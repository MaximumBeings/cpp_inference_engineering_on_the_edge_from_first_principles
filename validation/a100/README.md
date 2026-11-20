# A100 verification evidence

GPU and compiler configuration is in `gpu_environment.txt`. The CUDA transcript covers 60 shapes and 71 assertions. The sanitizer logs record actual device execution.

`linux_checks.json` records the initial sequential run. Its Appendix A failure was superseded by `verify_appendix_a_final.log` after installing build dependencies and fixing recognition of host-registered Arm emulation. `verify_chapter09_arm.log` includes the explicit QEMU run, and `verify_chapter32.log` checks CUDA on the A100. `embedded_examples.json` covers Chapters 2–6.

`cuda_no_device.txt` records execution with CUDA devices hidden.
