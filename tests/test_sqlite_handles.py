from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from evaluation import protocol


class TrackedConnection(sqlite3.Connection):
    closed_explicitly = False
    fail_on = None

    def close(self):
        self.closed_explicitly = True
        return super().close()

    def execute(self, statement, *args, **kwargs):
        if self.fail_on and statement.startswith(self.fail_on):
            raise sqlite3.OperationalError('injected database error')
        return super().execute(statement, *args, **kwargs)


class SQLiteHandleTests(unittest.TestCase):
    def setUp(self):
        self.connections = []
        self.real_connect = sqlite3.connect
        self.fail_on = None

        def connect(*args, **kwargs):
            kwargs['factory'] = TrackedConnection
            db = self.real_connect(*args, **kwargs)
            db.fail_on = self.fail_on
            self.connections.append(db)
            return db

        self.patch = mock.patch.object(protocol.sqlite3, 'connect', side_effect=connect)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def assert_closed(self):
        self.assertTrue(self.connections)
        self.assertTrue(all(db.closed_explicitly for db in self.connections))

    def test_allocator_constructor_releases_database_while_references_live(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nonce.sqlite'
            allocator = protocol.DurableNonceAllocator(path, 'generation', 7)
            self.assert_closed()
            path.unlink()
            self.assertIsNotNone(allocator)

    def test_replay_constructor_releases_database_while_references_live(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'replay.sqlite'
            receiver = protocol.DurableReplayWindow(path)
            self.assert_closed()
            path.unlink()
            self.assertIsNotNone(receiver)

    def test_nonce_success_and_exhaustion_close_all_connections(self):
        with tempfile.TemporaryDirectory() as folder:
            allocator = protocol.DurableNonceAllocator(Path(folder)/'nonce.sqlite', 'generation', 7, 2**64-1)
            allocator.take()
            with self.assertRaises(OverflowError):
                allocator.take()
            self.assert_closed()

    def test_replay_success_and_rejection_close_all_connections(self):
        with tempfile.TemporaryDirectory() as folder:
            key = protocol.auth_key(bytes(range(32)))
            receiver = protocol.DurableReplayWindow(Path(folder)/'replay.sqlite')
            packet = protocol.seal_header(protocol.encode_header(7, 0, 1), key)
            receiver.accept(packet, key)
            with self.assertRaises(ValueError):
                receiver.accept(packet, key)
            self.assert_closed()

    def test_initialization_failures_close_connections(self):
        self.fail_on = 'CREATE TABLE'
        with tempfile.TemporaryDirectory() as folder:
            for cls, args in ((protocol.DurableNonceAllocator, ('generation', 7)), (protocol.DurableReplayWindow, ())):
                with self.assertRaises(sqlite3.OperationalError):
                    cls(Path(folder)/'state.sqlite', *args)
                self.assert_closed()

    def test_pragma_failures_close_connections(self):
        self.fail_on = 'PRAGMA'
        with tempfile.TemporaryDirectory() as folder:
            for cls, args in ((protocol.DurableNonceAllocator, ('generation', 7)), (protocol.DurableReplayWindow, ())):
                with self.assertRaises(sqlite3.OperationalError):
                    cls(Path(folder)/'state.sqlite', *args)
                self.assert_closed()


if __name__ == '__main__':
    unittest.main()
