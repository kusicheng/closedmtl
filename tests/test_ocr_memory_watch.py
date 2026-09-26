"""Check that a short allocation missed by sampling survives in the peak counter."""

from pathlib import Path
import subprocess
import sys
import unittest

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training_scripts"))
from ocr_memory_watch import observe


class MemoryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=="win32", "Windows peak counters")
    def test_peak_is_available_after_exit(self):
        command="import time; time.sleep(0.5); data=bytearray(64*1024*1024); time.sleep(0.1)"
        child=subprocess.Popen([sys.executable, "-c", command])
        try:
            result=observe(child.pid, psutil.Process(child.pid).create_time())
            self.assertTrue(result["post_exit_counter_available"])
            self.assertGreaterEqual(result["ram_peak_bytes"], 64*1024*1024)
        finally:
            child.wait(timeout=10)


if __name__=="__main__":
    unittest.main()
