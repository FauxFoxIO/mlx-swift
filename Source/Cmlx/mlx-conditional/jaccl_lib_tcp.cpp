#include <TargetConditionals.h>

// See jaccl_conditional.cpp; keep the condition in sync.
#if TARGET_OS_OSX && __has_include(<infiniband/verbs.h>)
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/tcp.cpp"
#endif
