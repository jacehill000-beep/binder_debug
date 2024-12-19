# binder_debug
used in real device

This script is derived from pwndbg.

# command 

- binder 
- binder_proc addr 
- binder_node addr 


ex: 
```
(gdb) source <path>/binder.py
(gdb) binder_proc 0xffffffc06260e800
binder_proc PID 3677 (0xffffffc06260e800)
    is_dead: False
    tmp_ref: 0
    inner_lock:  LOCKED: 79 PENDING: 0 (0xffffffc06260ea38)
    outer_lock:  LOCKED: 3 PENDING: 0 (0xffffffc06260ea3c)
    waiting_threads [0]: EMPTY
    todo [0]: EMPTY
    threads [1]:
        * binder_thread PID 3721 (0xffffffc054892400)
            tmp_ref: 0
            looper_need_return: False
            process_todo: False
            is_dead: False
            todo [0]: EMPTY
            transaction_stack: NULL
    nodes [0]: EMPTY
    refs_by_node [1]:
        * binder_ref HANDLE 0 (0xffffffc01ef79980)
            strong: 1
            weak: 1
```

