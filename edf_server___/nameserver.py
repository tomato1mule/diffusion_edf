from typing import Optional, Union
import multiprocessing
import logging
import time

from beartype import beartype
from Pyro5.nameserver import start_ns, NameServerDaemon, BroadcastServer
from Pyro5.core import URI


class NameServer():
    nsUri: Optional[URI] = None
    daemon: Optional[NameServerDaemon] = None
    bcserver: Optional[BroadcastServer] = None
    ns_proc: Optional[multiprocessing.Process] = None
    running: multiprocessing.Value
    log: logging.Logger


    @beartype
    def __init__(self):
        self.log = logging.getLogger("NameServer")
        self.running = multiprocessing.Value('b', False)  # Boolean type
        self.init_nameserver()
        

    def init_nameserver(self, *args, **kwargs):
        self.nsUri, self.daemon, self.bcserver = start_ns(*args, **kwargs)
        self.ns_proc = multiprocessing.Process(target=self._ns_daemon_request_loop)
        self.ns_proc.start()

    def _ns_daemon_request_loop(self):
        with self.running.get_lock():
            self.running.value = True
        self.daemon.requestLoop(loopCondition=lambda :self.running.value)
        self.log.debug(f"Nameserver daemon loop exit")

    def close(self, timeout = 5.) -> bool:
        self.log.warning(f"Closing Nameserver... (Timeout: {timeout} sec)")
        with self.running.get_lock():
            self.running.value = False
        init_time = time.time()
        self.ns_proc.join(timeout=timeout)
        if self.ns_proc.is_alive():
            self.log.error(f"Nameserver daemon loop process not cleanly closed in {time.time() - init_time} seconds.")
            return False
        else:
            self.daemon.close()
            if self.bcserver is not None:
                self.bcserver.close()
            self.ns_proc.close()
            self.log.warning(f"Nameserver daemon loop process cleanly closed in {time.time() - init_time} seconds.")
            return True
