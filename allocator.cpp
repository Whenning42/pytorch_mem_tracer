#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

std::string current_line = "untraced";

size_t cur_mem = 0;
size_t peak_mem = 0;
size_t total_alloc = 0;
size_t allocations = 0;

std::unordered_map<std::string, size_t> string_handles;
std::vector<std::string> strings;


size_t intern(std::string str) {
  auto handle = string_handles.find(str);
  if (handle == string_handles.end()) {
    size_t next = strings.size();
    bool ignore;
    std::tie(handle, ignore) = string_handles.insert({str, next});
    strings.push_back(str);
  }
  return handle->second;
}

std::string get_str(size_t handle) {
  return strings[handle];
}

struct Allocation {
  size_t size;
  size_t location_handle;
};

struct AllocationStats {
  size_t current_bytes = 0;
  size_t total_bytes = 0;
  size_t current_allocs = 0;
  size_t total_allocs = 0;

  std::unordered_map<void*, Allocation> live_allocations;
};

AllocationStats mem_stats;
AllocationStats peak_stats;

std::string fmt_readable(size_t bytes) {
  double bytes_d = bytes;
  const char* units[] = {"", "KB", "MB", "GB"};
  int i = 0;
  for (; i < 3; ++i) {
    if (bytes_d < 1024) {
      break;
    }

    bytes_d /= 1024;
  }

  std::stringstream ss;
  const int digits = 4;
  int leading_digits = static_cast<int>(std::floor(std::log10(std::abs(bytes_d)))) + 1;
  ss << std::fixed << std::setprecision(4 - leading_digits) << bytes_d << " " << units[i];
  return ss.str();
}

extern "C" {

void push_line(const char *line) {
  if (line) {
    current_line = line;
  }
}


void report_stats() {
  if (mem_stats.current_bytes > peak_stats.current_bytes) {
    peak_stats = mem_stats;
  }

  std::cout << "Max mem: " << fmt_readable(peak_stats.current_bytes) << std::endl;
  std::cout << "Total mem allocated: " << fmt_readable(peak_stats.total_bytes) << std::endl;

  std::unordered_map<size_t, size_t> bytes_by_line_handle;
  std::unordered_map<size_t, std::vector<size_t>> sizes_by_line_handle;
  for (const auto& [ptr, alloc] : peak_stats.live_allocations) {
    size_t handle = alloc.location_handle;
    bytes_by_line_handle[handle] += alloc.size;
    sizes_by_line_handle[handle].push_back(alloc.size);
  }

  // Print the allocation locations sorted from most to least bytes.
  std::vector<std::pair<size_t, size_t>> bytes_by_line_vec =
      std::vector<std::pair<size_t, size_t>>(bytes_by_line_handle.begin(), bytes_by_line_handle.end());

  std::sort(bytes_by_line_vec.begin(), bytes_by_line_vec.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::cout << "Mem, Cumulative Mem, Median, Count, Location" << std::endl;
  size_t cumulative_bytes = 0;
  for (const auto [line_handle, bytes] : bytes_by_line_vec) {
    cumulative_bytes += bytes;

    // TODO: Hide median and count behind options.
    auto& line_allocs = sizes_by_line_handle[line_handle];
    std::sort(line_allocs.begin(), line_allocs.end());
    size_t median_size = line_allocs[line_allocs.size() / 2];
    size_t count = line_allocs.size();

    std::cout << fmt_readable(bytes) << "  " << fmt_readable(cumulative_bytes) << "  " << fmt_readable(median_size) << "  " << count << "  " << get_str(line_handle) << std::endl;
  }
}

void* malloc_impl(size_t size, int device, cudaStream_t stream) {
  void *ptr;
  cudaMalloc(&ptr, size);

  mem_stats.current_bytes += size;
  mem_stats.total_bytes += size;
  mem_stats.current_allocs++;
  mem_stats.total_allocs++;
  
  size_t loc_handle = intern(current_line);
  mem_stats.live_allocations.insert({ptr, Allocation{.size=size, .location_handle=loc_handle}});

  return ptr;
}

void free_impl(void* ptr, size_t size, int device, cudaStream_t stream) {
  mem_stats.current_bytes -= size;
  mem_stats.current_allocs--;
  mem_stats.live_allocations.erase(ptr);

  if (mem_stats.current_bytes > peak_stats.current_bytes) {
    peak_stats = mem_stats;
  }
  cudaFree(ptr);
}

} // extern "C"

