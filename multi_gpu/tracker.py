
import os
import psutil
from threading import Thread
import time
import numpy as np
from torch.cuda import memory_allocated
from torch.cuda import memory_reserved
from pynvml import *


class MemoryTracker:

    def __init__(self, process_name, device, track_interval=0.2):
        self.process_name = process_name
        self.device = device
        self.deviceID = int(self.device[-1])
        self.track_interval = track_interval
        self.running = False
        self.ram_usage = []
        self.torch_allocated_vram = [] #NEW 
        self.torch_reserved_vram  = [] # NEW
        self.total_process_vram = [] # NEW
        self.cpu_usage = []
        self.proc_cpu_usage = [] 
        self.gpu_usage = []
        self.gpu_temp = []
        self.gpu_clock = []
        self.cpu_clock = []
        self.timestamps = []
        # Context switch tracking variables
        self.vol_ctx_switches = []
        self.invol_ctx_switches = []
        self.vol_ctx_rate = []
        self.invol_ctx_rate = []
        
        #PCIe variables
        self.pcie_tx_mb = []
        self.pcie_rx_mb = []
        self.prev_pcie_tx = 0
        self.prev_pcie_rx = 0
        
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.peak_ram = 0
        self.peak_torch_vram = 0
        self.peak_vram = 0
        # Track previous context switch values for rate calculation
        self.prev_vol_ctx = 0
        self.prev_invol_ctx = 0
        self.prev_ctx_time = time.time()

        # Init NVML
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(self.deviceID)  # GPU 0
            self.gpu_available = True
            
            # Initialize PCIe counters if available
            try:
                self.prev_pcie_tx = nvmlDeviceGetPcieThroughput(
                    self.handle, NVML_PCIE_UTIL_TX_BYTES)
                self.prev_pcie_rx = nvmlDeviceGetPcieThroughput(
                    self.handle, NVML_PCIE_UTIL_RX_BYTES)
            except:
                pass
                
        except:
            self.gpu_available = False
            
        # Try to import PyTorch for CUDA metrics
        try:
            import torch
            self.torch_available = torch.cuda.is_available()
            print("torch available")
        except ImportError:
            self.torch_available = False
            print("torch not available")

        self.power_mw = []

    def start(self):
        self.running = True
        self.thread = Thread(target=self._track, daemon=True)
        self.thread.start()

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        self.thread.join()
        if self.gpu_available:
            nvmlShutdown()
        return {
            'process': self.process_name,
            'pid': self.pid,
            'gpu': {
                
                'avg_gpu_usage': float(np.mean(self.gpu_usage)) if self.gpu_usage else 0,
                'max_gpu_usage': float(np.max(self.gpu_usage)) if self.gpu_usage else 0,
                'avg_gpu_clock_mhz': float(np.mean(self.gpu_clock)) if self.gpu_clock else 0,
                'max_gpu_clock_mhz': float(np.max(self.gpu_clock)) if self.gpu_clock else 0,
                'avg_gpu_temp': float(np.mean(self.gpu_temp)) if self.gpu_temp else 0,

            },
            'vram': {
                'avg_torch_allocated_vram': float(np.mean(self.torch_allocated_vram)) if self.torch_allocated_vram else 0,
                'max_torch_allocated_vram': float(np.max(self.torch_allocated_vram)) if self.torch_allocated_vram else 0,
                'min_torch_allocated_vram': float(np.min(self.torch_allocated_vram)) if self.torch_allocated_vram else 0,
                'std_torch_allocated_vram': float(np.std(self.torch_allocated_vram)) if self.torch_allocated_vram else 0,
                
                'avg_torch_reserved_vram': float(np.mean(self.torch_reserved_vram)) if self.torch_reserved_vram else 0,
                'max_torch_reserved_vram': float(np.max(self.torch_reserved_vram)) if self.torch_reserved_vram else 0,
                'min_torch_reserved_vram': float(np.min(self.torch_reserved_vram)) if self.torch_reserved_vram else 0,
                'std_torch_reserved_vram': float(np.std(self.torch_reserved_vram)) if self.torch_reserved_vram else 0,

                'avg_total_process_vram': float(np.mean(self.total_process_vram)) if self.total_process_vram else 0,
                'max_total_process_vram': float(np.max(self.total_process_vram)) if self.total_process_vram else 0,
                'min_total_process_vram': float(np.min(self.total_process_vram)) if self.total_process_vram else 0,
                'std_total_process_vram': float(np.std(self.total_process_vram)) if self.total_process_vram else 0,

                'peak_torch_vram': self.peak_torch_vram,
                'peak_process_vram': self.peak_vram,

            },
            'cpu': {
                'avg_cpu_usage': float(np.mean(self.cpu_usage)),
                'avg_proc_cpu_usage': float(np.mean(self.proc_cpu_usage)),
                'max_proc_cpu_usage': float(np.max(self.proc_cpu_usage)),
                'avg_cpu_clock_mhz': float(np.mean(self.cpu_clock)) if self.cpu_clock else 0,
                # Context switch stats
                'total_vol_ctx_switches': int(self.vol_ctx_switches[-1]) if self.vol_ctx_switches else 0,
                'total_invol_ctx_switches': int(self.invol_ctx_switches[-1]) if self.invol_ctx_switches else 0,
                'avg_vol_ctx_rate': float(np.mean(self.vol_ctx_rate)) if self.vol_ctx_rate else 0,
                'avg_invol_ctx_rate': float(np.mean(self.invol_ctx_rate)) if self.invol_ctx_rate else 0,
            },
            'ram': {
                    
                'avg_ram_mb': float(np.mean(self.ram_usage)),
                'max_ram_mb': float(np.max(self.ram_usage)),
                'min_ram_mb': float(np.min(self.ram_usage)),
                'std_ram_mb': float(np.std(self.ram_usage)),
                'peak_ram_mb': self.peak_ram,
            },

            # CUDA metrics
            'avg_pcie_tx_mb': float(np.mean(self.pcie_tx_mb)) if self.pcie_tx_mb else 0,
            'avg_pcie_rx_mb': float(np.mean(self.pcie_rx_mb)) if self.pcie_rx_mb else 0,
            # Timelines
            'sys_timelines': {
                'ram_timeline': list(zip(self.timestamps, self.ram_usage)),
                'vram_timeline': list(zip(self.timestamps, self.total_process_vram)),
                'torch_vram_allocated_timeline': list(zip(self.timestamps, self.torch_allocated_vram)),
                'torch_vram_reserved_timeline': list(zip(self.timestamps, self.torch_reserved_vram)),
                'cpu_usage_timeline': list(zip(self.timestamps, self.cpu_usage)),
                'proc_cpu_usage_timeline': list(zip(self.timestamps, self.proc_cpu_usage)),
                'cpu_clock_timeline': list(zip(self.timestamps, self.cpu_clock)),
                'gpu_usage_timeline': list(zip(self.timestamps, self.gpu_usage)),
                'gpu_clock_timeline': list(zip(self.timestamps, self.gpu_clock)),
                'gpu_temp_timeline': list(zip(self.timestamps, self.gpu_temp)),
                # Context switch timelines
                'vol_ctx_switches_timeline': list(zip(self.timestamps, self.vol_ctx_switches)),
                'invol_ctx_switches_timeline': list(zip(self.timestamps, self.invol_ctx_switches)),
                'vol_ctx_rate_timeline': list(zip(self.timestamps, self.vol_ctx_rate)),
                'invol_ctx_rate_timeline': list(zip(self.timestamps, self.invol_ctx_rate)),
                'power_mw_timeline' :  list(zip(self.timestamps, self.power_mw))
                }
            }

    def _track(self):
        start_time = time.time()
        self.prev_ctx_time = start_time

        while self.running:
            timestamp = time.time() - start_time
            self.timestamps.append(timestamp)

            # RAM
            ram = self.process.memory_info().rss / (1024 * 1024)
            self.ram_usage.append(ram)
            self.peak_ram = max(self.peak_ram, ram)

            # VRAM
            try:
                torch_alloc_vram = memory_allocated(device=self.deviceID) / (1024 * 1024)
                torch_reserved_vram = memory_reserved(device=self.deviceID) / (1024 * 1024)
                self.torch_allocated_vram.append(torch_alloc_vram)
                self.torch_reserved_vram.append(torch_reserved_vram)
                
                self.peak_torch_vram = max(self.peak_torch_vram, torch_alloc_vram)
            except:
                self.torch_allocated_vram.append(0)
                self.torch_reserved_vram.append(0) 

            # CPU Utilization overall
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu_percent)

            p_cpu_percercent = self.process.cpu_percent()
            self.proc_cpu_usage.append(p_cpu_percercent)

            try:
                freqs = psutil.cpu_freq(percpu=True)
                avg_clock = np.mean([f.current for f in freqs])
                self.cpu_clock.append(avg_clock)
            except:
                self.cpu_clock.append(0)


            # GPU Stats
            if self.gpu_available:
                try:
                    util = nvmlDeviceGetUtilizationRates(self.handle)
                    self.gpu_usage.append(util.gpu)

                    power_mw = nvmlDeviceGetPowerUsage(self.handle)
                    self.power_mw.append(power_mw)

                    clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_GRAPHICS)
                    self.gpu_clock.append(clock)

                    temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
                    self.gpu_temp.append(temp)

                    procs = nvmlDeviceGetComputeRunningProcesses(self.handle)
                    for p in procs:
                        if p.pid == self.pid:
                            t_used = p.usedGpuMemory / 1024**2
                            self.total_process_vram.append(t_used)
                            self.peak_vram = max(self.peak_vram, t_used)

                    
                    # PCIe transfers monitoring
                    try:
                        current_tx = nvmlDeviceGetPcieThroughput(self.handle, NVML_PCIE_UTIL_TX_BYTES)
                        current_rx = nvmlDeviceGetPcieThroughput(self.handle, NVML_PCIE_UTIL_RX_BYTES)
                    
                        current_tx_mbs = current_tx / (1024*1024)
                        current_rx_mbs = current_rx / (1024*1024)

                        self.pcie_tx_mb.append(current_tx_mbs)
                        self.pcie_rx_mb.append(current_rx_mbs)
                        
            
                    except:
                        self.pcie_tx_mb.append(0)
                        self.pcie_rx_mb.append(0)
                        
                except:
                    self.gpu_usage.append(0)
                    self.gpu_clock.append(0)
                    self.gpu_temp.append(0)

                    self.pcie_tx_mb.append(0)
                    self.pcie_rx_mb.append(0)
            else:
                self.gpu_usage.append(0)
                self.gpu_clock.append(0)
                self.gpu_temp.append(0)

                self.pcie_tx_mb.append(0)
                self.pcie_rx_mb.append(0)
                

            # CONTEXT SWITCHES
            try:

                ctx_switches = self.process.num_ctx_switches()
                vol_ctx = ctx_switches.voluntary
                invol_ctx = ctx_switches.involuntary
                
                self.vol_ctx_switches.append(vol_ctx)
                self.invol_ctx_switches.append(invol_ctx)
                
                current_time = time.time()
                time_diff = current_time - self.prev_ctx_time
                
                if time_diff > 0 and hasattr(self, 'prev_vol_ctx'):
                    vol_rate = (vol_ctx - self.prev_vol_ctx) / time_diff
                    invol_rate = (invol_ctx - self.prev_invol_ctx) / time_diff
                    self.vol_ctx_rate.append(vol_rate)
                    self.invol_ctx_rate.append(invol_rate)
                else:
                    self.vol_ctx_rate.append(0)
                    self.invol_ctx_rate.append(0)
                
                self.prev_vol_ctx = vol_ctx
                self.prev_invol_ctx = invol_ctx
                self.prev_ctx_time = current_time
                
            except Exception as e:
                self.vol_ctx_switches.append(0)
                self.invol_ctx_switches.append(0)
                self.vol_ctx_rate.append(0)
                self.invol_ctx_rate.append(0)
                print(f"Error tracking context switches: {e}")

            time.sleep(self.track_interval)