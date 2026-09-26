"""Retain a Windows process handle to read peak RAM after the process exits."""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path

import psutil


class Counters(ctypes.Structure):
    _fields_=[("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]+[
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


def observe(pid, create_time):
    if os.name!="nt":
        raise RuntimeError("Windows process memory counters are required")
    process=psutil.Process(pid)
    if abs(process.create_time()-create_time)>0.001:
        raise RuntimeError("PID was reused; refusing to observe a different process")
    kernel=ctypes.WinDLL("kernel32", use_last_error=True)
    api=ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.GetProcessMemoryInfo.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x100410, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    peak=private_peak=0
    try:
        while True:
            state=kernel.WaitForSingleObject(handle, 1000)
            if state not in (0, 258):
                raise ctypes.WinError(ctypes.get_last_error())
            counters=Counters()
            counters.cb=ctypes.sizeof(counters)
            success=api.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            if success:
                peak=max(peak, counters.PeakWorkingSetSize)
                private_peak=max(private_peak, counters.PeakPagefileUsage)
            if state==0:
                return {"pid":pid, "create_time":create_time,
                        "ram_peak_method":"windows_retained_process_handle",
                        "ram_peak_bytes":peak, "peak_pagefile_usage_bytes":private_peak,
                        "post_exit_counter_available":bool(success), "process_exited":True}
            if not success:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--create-time", type=float, required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    result=observe(args.pid, args.create_time)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
