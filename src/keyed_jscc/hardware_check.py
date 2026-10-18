from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
from .config import project_root
import platform
import re
import shutil
import struct
import subprocess
import sys

def _safe_error(error):
    message = str(error).splitlines()[0] if str(error) else type(error).__name__
    for key in ('USERPROFILE', 'HOME'):
        value = os.environ.get(key)
        if value:
            message = message.replace(value, '<user folder>')
    message = re.sub('[A-Za-z]:[\\\\/][^\\r\\n]*', '<local path>', message)
    message = re.sub('/(?:Users|home)/[^/\\s]+', '/<user folder>', message)
    return {'type': type(error).__name__, 'message': message[:600]}

def _physical_memory():
    if platform.system() == 'Windows':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total_phys', ctypes.c_ulonglong), ('available_phys', ctypes.c_ulonglong), ('total_page', ctypes.c_ulonglong), ('available_page', ctypes.c_ulonglong), ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong), ('available_extended', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys)
    return None

def _nvidia_summary():
    candidate = shutil.which('nvidia-smi')
    if candidate is None and platform.system() == 'Windows':
        fallback = Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'System32/nvidia-smi.exe'
        if fallback.exists():
            candidate = str(fallback)
    if candidate is None:
        return {'available': False, 'gpus': []}
    try:
        completed = subprocess.run([candidate, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20, check=True)
        rows = []
        for row in csv.reader(io.StringIO(completed.stdout)):
            if len(row) != 3:
                continue
            rows.append({'name': row[0].strip(), 'driver_version': row[1].strip(), 'memory_total_mib': float(row[2].strip())})
        return {'available': True, 'gpus': rows}
    except Exception as error:
        return {'available': False, 'gpus': [], 'error': _safe_error(error)}

def inspect_hardware():
    report = {'schema_version': 1, 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'system': {'os': platform.system(), 'os_release': platform.release(), 'architecture': platform.machine(), 'python_version': platform.python_version(), 'python_bits': struct.calcsize('P') * 8, 'logical_cpu_count': os.cpu_count()}, 'physical_memory_bytes': _physical_memory(), 'nvidia_smi': _nvidia_summary(), 'package_versions': {}, 'cuda_available': False, 'cuda_smoke_test': {'passed': False}, 'ready_for_gpu_training': False}
    for package in ('torch', 'numpy', 'Pillow', 'cryptography', 'scipy', 'scikit-learn', 'tqdm'):
        try:
            report['package_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch
        report['torch'] = {'version': torch.__version__, 'built_cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(), 'compiled_cuda_architectures': torch.cuda.get_arch_list()}
        report['cuda_available'] = bool(torch.cuda.is_available())
        if not report['cuda_available']:
            report['next_step'] = 'CUDA unavailable'
            return report
        report['cuda_devices'] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report['cuda_devices'].append({'index': index, 'name': properties.name, 'total_memory_bytes': int(properties.total_memory), 'compute_capability': [int(properties.major), int(properties.minor)], 'multiprocessor_count': int(properties.multi_processor_count)})
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        report['cuda_memory_before_probe'] = {'free_bytes': int(free), 'total_bytes': int(total)}
        report['cuda_bfloat16_supported'] = bool(torch.cuda.is_bf16_supported())
        product = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
        if not bool(torch.isfinite(product).all()):
            raise RuntimeError('CUDA matrix multiplication returned nonfinite values')
        model = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
        inputs = torch.randn(4, 3, 16, 16, device=device)
        fp32_loss = model(inputs).square().mean()
        fp32_loss.backward()
        if not all((bool(torch.isfinite(p.grad).all()) for p in model.parameters())):
            raise RuntimeError('FP32 convolution gradients are nonfinite')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        scaler = torch.amp.GradScaler('cuda')
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            amp_loss = model(inputs).square().mean()
        scaler.scale(amp_loss).backward()
        scaler.unscale_(optimizer)
        if not all((bool(torch.isfinite(p.grad).all()) for p in model.parameters())):
            raise RuntimeError('AMP convolution gradients are nonfinite')
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        operations = ['FP32 matrix multiplication', 'FP32 convolution forward/backward', 'FP16 AMP convolution forward/backward and optimizer step']
        bf16_loss = None
        if report['cuda_bfloat16_supported']:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                bf16_loss = model(inputs).square().mean()
            bf16_loss.backward()
            if not all((bool(torch.isfinite(p.grad).all()) for p in model.parameters())):
                raise RuntimeError('BF16 AMP convolution gradients are nonfinite')
            optimizer.step()
            torch.cuda.synchronize(device)
            operations.append('BF16 AMP convolution forward/backward and optimizer step')
        report['cuda_smoke_test'] = {'passed': True, 'device_index': 0, 'operations': operations, 'fp32_loss': float(fp32_loss.detach().cpu()), 'amp_loss': float(amp_loss.detach().cpu()), 'bf16_amp_loss': None if bf16_loss is None else float(bf16_loss.detach().cpu()), 'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device))}
        report['ready_for_gpu_training'] = True
        report['next_step'] = 'GPU checks passed'
    except Exception as error:
        report['cuda_smoke_test'] = {'passed': False, 'error': _safe_error(error)}
        report['next_step'] = 'CUDA preflight failed'
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description='Check the Python environment and CUDA training operations.')
    parser.add_argument('--output', type=Path, default=project_root() / 'outputs/hardware_report.json')
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args(argv)
    print('Checking GPU', flush=True)
    report = inspect_hardware()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for device in report.get('cuda_devices', []):
        print(f"GPU: {device['name']}; VRAM {device['total_memory_bytes'] / 2 ** 30:.1f} GiB; compute capability {device['compute_capability'][0]}.{device['compute_capability'][1]}", flush=True)
    print(report['next_step'], flush=True)
    error = report.get('cuda_smoke_test', {}).get('error')
    if error:
        print(f"Check detail: {error['message']}", flush=True)
    print(f'Hardware report saved as {args.output.name}.', flush=True)
    return 0 if report['ready_for_gpu_training'] or not args.require_cuda else 2
if __name__ == '__main__':
    sys.exit(main())
