#include <TargetConditionals.h>

// JACCL (RDMA over Thunderbolt) needs <infiniband/verbs.h>, macOS SDK 26.2+ only;
// librdma.dylib is dlopen'd at runtime. Elsewhere use the stub.
#if TARGET_OS_OSX && __has_include(<infiniband/verbs.h>)
#include "../mlx/mlx/distributed/jaccl/jaccl.cpp"
#else
#include "../mlx/mlx/distributed/jaccl/no_jaccl.cpp"
#endif
