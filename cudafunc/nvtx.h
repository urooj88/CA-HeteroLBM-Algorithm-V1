#pragma once
#include <stdio.h>

// Dummy NVTX header for CUDA 13+ where nvToolsExt.h is not available.
// Keeps the same macros (RANGE_PUSH / RANGE_POP) so the rest of the code compiles,
// but they expand to no-ops (do nothing).

// Color definitions (kept for compatibility)
#define CLR_BLUE    0xFF0000FF
#define CLR_MAGENTA 0xFFFF00FF
#define CLR_PURPLE  0xFF800080

// Macros now expand to nothing
#define RANGE_PUSH(range_title,range_color) {}
#define RANGE_POP {}

