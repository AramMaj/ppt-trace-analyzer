# How We Test

No automated tests unfortunatly.

We generate trace files with torchtitan on our supervisor's GPU server, copy them down, and visually inspect the output in Perfetto UI.
It's a bit messy: we look for phase bars, flow arrows, and bottleneck markers that look plausable and call it a day.

If you want to run the code, grab a trace from the `traces/` directory or create your own