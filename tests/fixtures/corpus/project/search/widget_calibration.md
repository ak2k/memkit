---
name: widget_calibration
description: A firmware flash clears a widget's stored trim table, so the zero point has to be taught again before the readings mean anything.
type: reference
---

# Widget calibration

A firmware flash clears the trim table, so a widget reports its old
zero until it is taught a new one.

## Teaching the zero

Hold the widget at the reference stop and issue the teach command;
the trim table is rewritten from that one sample.
