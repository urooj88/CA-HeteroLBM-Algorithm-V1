#ifndef RUNTIME_RESOURCE_CONTROL_H
#define RUNTIME_RESOURCE_CONTROL_H

#include <hwloc.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace rrc {

enum class BindMode {
    Off,
    Auto,
    Manual
};

enum class CoreKind {
    Logical,
    Physical
};

struct RuntimeResourceOptions {
    int cpu_threads = -1;
    BindMode bind_mode = BindMode::Auto;
    CoreKind core_kind = CoreKind::Logical;
    bool print_topology = false;
    bool print_binding = false;
    bool print_monitor = false;
    int monitor_every = 1000;
    int cpu_numa = -1;

    std::vector<int> gpu_host_cores;
    std::vector<int> cpu_cores;
};

struct TopologySummary {
    int logical_cores = 0;
    int physical_cores = 0;
    int numa_nodes = 0;
    int packages = 0;
    bool hyperthreading = false;

    std::vector<int> logical_os_ids;
    std::vector<int> physical_os_ids;
    std::vector<int> logical_to_physical_os;
    std::vector<std::vector<int>> numa_logical_cores;
};

struct BindingDecision {
    std::vector<int> selected_logical_cores;
    std::vector<int> selected_physical_cores;
    int selected_numa = -1;
    std::string source;
};

static inline std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

static inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static inline bool parse_bool_string(const std::string& s) {
    const std::string v = to_lower(trim(s));
    return (v == "1" || v == "on" || v == "true" || v == "yes");
}

static inline std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }

        const std::size_t dash = token.find('-');
        if (dash != std::string::npos) {
            const int a = std::stoi(trim(token.substr(0, dash)));
            const int b = std::stoi(trim(token.substr(dash + 1)));
            if (a <= b) {
                for (int x = a; x <= b; ++x) out.push_back(x);
            } else {
                for (int x = a; x >= b; --x) out.push_back(x);
            }
        } else {
            out.push_back(std::stoi(token));
        }
    }

    return out;
}

static inline const char* bind_mode_to_cstr(BindMode m) {
    switch (m) {
        case BindMode::Off: return "off";
        case BindMode::Auto: return "auto";
        case BindMode::Manual: return "manual";
    }
    return "unknown";
}

static inline const char* core_kind_to_cstr(CoreKind k) {
    switch (k) {
        case CoreKind::Logical: return "logical";
        case CoreKind::Physical: return "physical";
    }
    return "unknown";
}

static inline RuntimeResourceOptions parse_runtime_resource_options(int argc, char** argv) {
    RuntimeResourceOptions opt;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) {
                std::ostringstream oss;
                oss << "Missing value for " << name;
                throw std::runtime_error(oss.str());
            }
        };

        if (arg == "--cpu-threads") {
            need_value("--cpu-threads");
            opt.cpu_threads = std::stoi(argv[++i]);
        } else if (arg == "--bind") {
            need_value("--bind");
            const std::string v = to_lower(argv[++i]);
            if (v == "off") opt.bind_mode = BindMode::Off;
            else if (v == "auto") opt.bind_mode = BindMode::Auto;
            else if (v == "manual") opt.bind_mode = BindMode::Manual;
            else throw std::runtime_error("Invalid value for --bind. Use off|auto|manual");
        } else if (arg == "--core-type") {
            need_value("--core-type");
            const std::string v = to_lower(argv[++i]);
            if (v == "logical") opt.core_kind = CoreKind::Logical;
            else if (v == "physical") opt.core_kind = CoreKind::Physical;
            else throw std::runtime_error("Invalid value for --core-type. Use logical|physical");
        } else if (arg == "--gpu-host-cores") {
            need_value("--gpu-host-cores");
            opt.gpu_host_cores = parse_int_list(argv[++i]);
        } else if (arg == "--cpu-cores") {
            need_value("--cpu-cores");
            opt.cpu_cores = parse_int_list(argv[++i]);
        } else if (arg == "--cpu-numa") {
            need_value("--cpu-numa");
            opt.cpu_numa = std::stoi(argv[++i]);
        } else if (arg == "--print-topology") {
            if (i + 1 < argc && argv[i + 1][0] != '-') opt.print_topology = parse_bool_string(argv[++i]);
            else opt.print_topology = true;
        } else if (arg == "--print-binding") {
            if (i + 1 < argc && argv[i + 1][0] != '-') opt.print_binding = parse_bool_string(argv[++i]);
            else opt.print_binding = true;
        } else if (arg == "--print-monitor") {
            if (i + 1 < argc && argv[i + 1][0] != '-') opt.print_monitor = parse_bool_string(argv[++i]);
            else opt.print_monitor = true;
        } else if (arg == "--monitor-every") {
            need_value("--monitor-every");
            opt.monitor_every = std::max(1, std::stoi(argv[++i]));
        }
    }

    if (opt.cpu_threads == 0) opt.cpu_threads = -1;
    return opt;
}

class TopologyHandle {
public:
    TopologyHandle() : topology_(nullptr) {
        if (hwloc_topology_init(&topology_) != 0) {
            throw std::runtime_error("hwloc_topology_init failed");
        }
        if (hwloc_topology_load(topology_) != 0) {
            hwloc_topology_destroy(topology_);
            topology_ = nullptr;
            throw std::runtime_error("hwloc_topology_load failed");
        }
    }

    ~TopologyHandle() {
        if (topology_ != nullptr) {
            hwloc_topology_destroy(topology_);
            topology_ = nullptr;
        }
    }

    TopologyHandle(const TopologyHandle&) = delete;
    TopologyHandle& operator=(const TopologyHandle&) = delete;

    hwloc_topology_t get() const { return topology_; }

private:
    hwloc_topology_t topology_;
};

static inline int find_numa_for_logical_core(hwloc_topology_t topo, int pu_os_index) {
    const int nnuma = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    if (nnuma <= 0) return -1;

    for (int n = 0; n < nnuma; ++n) {
        hwloc_obj_t node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, n);
        if (!node || !node->cpuset) continue;

        hwloc_bitmap_t single = hwloc_bitmap_alloc();
        hwloc_bitmap_only(single, pu_os_index);
        const bool contained = hwloc_bitmap_isincluded(single, node->cpuset);
        hwloc_bitmap_free(single);

        if (contained) {
            return static_cast<int>(node->os_index);
        }
    }
    return -1;
}

static inline TopologySummary query_topology_summary(hwloc_topology_t topo) {
    TopologySummary s;

    s.logical_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
    s.physical_cores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
    s.numa_nodes = std::max(0, hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE));
    s.packages = std::max(0, hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PACKAGE));
    s.hyperthreading = (s.logical_cores > s.physical_cores);

    s.logical_os_ids.reserve(std::max(0, s.logical_cores));
    s.physical_os_ids.reserve(std::max(0, s.physical_cores));
    s.logical_to_physical_os.reserve(std::max(0, s.logical_cores));

    for (int i = 0; i < s.logical_cores; ++i) {
        hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, i);
        if (!pu) continue;

        s.logical_os_ids.push_back(static_cast<int>(pu->os_index));

        hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu);
        if (core) s.logical_to_physical_os.push_back(static_cast<int>(core->os_index));
        else s.logical_to_physical_os.push_back(-1);
    }

    for (int i = 0; i < s.physical_cores; ++i) {
        hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, i);
        if (core) s.physical_os_ids.push_back(static_cast<int>(core->os_index));
    }

    if (s.numa_nodes > 0) {
        s.numa_logical_cores.resize(s.numa_nodes);
        for (int i = 0; i < s.logical_cores; ++i) {
            hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, i);
            if (!pu) continue;
            const int pu_os = static_cast<int>(pu->os_index);
            const int numa = find_numa_for_logical_core(topo, pu_os);
            if (numa >= 0 && numa < s.numa_nodes) {
                s.numa_logical_cores[numa].push_back(pu_os);
            }
        }
    }

    return s;
}

static inline std::vector<int> physical_core_to_logical_cores(hwloc_topology_t topo, int physical_core_os) {
    std::vector<int> out;
    const int ncores = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
    for (int i = 0; i < ncores; ++i) {
        hwloc_obj_t core = hwloc_get_obj_by_type(topo, HWLOC_OBJ_CORE, i);
        if (!core || static_cast<int>(core->os_index) != physical_core_os) continue;

        for (unsigned c = 0; c < core->arity; ++c) {
            hwloc_obj_t child = core->children[c];
            if (child && child->type == HWLOC_OBJ_PU) {
                out.push_back(static_cast<int>(child->os_index));
            }
        }
        break;
    }
    std::sort(out.begin(), out.end());
    return out;
}

static inline std::vector<int> logical_to_physical_list(hwloc_topology_t topo, const std::vector<int>& logical_cores) {
    std::set<int> uniq;
    for (int lc : logical_cores) {
        const int npus = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
        for (int i = 0; i < npus; ++i) {
            hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, i);
            if (!pu || static_cast<int>(pu->os_index) != lc) continue;

            hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu);
            if (core) uniq.insert(static_cast<int>(core->os_index));
            break;
        }
    }
    return std::vector<int>(uniq.begin(), uniq.end());
}

static inline void print_topology_summary(std::ostream& os, hwloc_topology_t topo) {
    const TopologySummary s = query_topology_summary(topo);
    os << "Runtime topology summary\n";
    os << "  logical cores : " << s.logical_cores << "\n";
    os << "  physical cores: " << s.physical_cores << "\n";
    os << "  NUMA nodes    : " << s.numa_nodes << "\n";
    os << "  packages      : " << s.packages << "\n";
    os << "  hyperthreading: " << (s.hyperthreading ? "on" : "off") << "\n";
    os << "  logical -> physical map:\n";
    for (std::size_t i = 0; i < s.logical_os_ids.size(); ++i) {
        os << "    lcore " << s.logical_os_ids[i] << " -> pcore "
           << s.logical_to_physical_os[i] << "\n";
    }

    if (s.numa_nodes > 0) {
        os << "  NUMA logical core lists:\n";
        for (int n = 0; n < s.numa_nodes; ++n) {
            os << "    numa " << n << ": ";
            for (std::size_t i = 0; i < s.numa_logical_cores[n].size(); ++i) {
                os << s.numa_logical_cores[n][i] << (i + 1 == s.numa_logical_cores[n].size() ? "" : " ");
            }
            os << "\n";
        }
    }
}

static inline std::vector<int> get_numa_logical_cores(hwloc_topology_t topo, int numa_index) {
    const TopologySummary s = query_topology_summary(topo);
    if (numa_index < 0 || numa_index >= static_cast<int>(s.numa_logical_cores.size())) {
        return {};
    }
    return s.numa_logical_cores[numa_index];
}

static inline std::vector<int> get_interleaved_numa_logical_cores(hwloc_topology_t topo) {
    const TopologySummary s = query_topology_summary(topo);
    if (s.numa_nodes <= 1 || s.numa_logical_cores.empty()) {
        return s.logical_os_ids;
    }

    std::vector<std::vector<int>> per_numa = s.numa_logical_cores;
    std::vector<std::size_t> pos(per_numa.size(), 0);
    std::vector<int> out;
    out.reserve(s.logical_os_ids.size());

    bool progressed = true;
    while (progressed) {
        progressed = false;
        for (std::size_t n = 0; n < per_numa.size(); ++n) {
            if (pos[n] < per_numa[n].size()) {
                out.push_back(per_numa[n][pos[n]++]);
                progressed = true;
            }
        }
    }

    if (out.empty()) return s.logical_os_ids;
    return out;
}

static inline std::string slurp_command_output(const std::string& cmd) {
    std::array<char, 512> buffer{};
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return result;

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

static inline std::vector<int> expand_cpu_affinity_token(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty() || token == "N/A") continue;

        const std::size_t dash = token.find('-');
        if (dash != std::string::npos) {
            const int a = std::stoi(trim(token.substr(0, dash)));
            const int b = std::stoi(trim(token.substr(dash + 1)));
            for (int x = a; x <= b; ++x) out.push_back(x);
        } else {
            out.push_back(std::stoi(token));
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static inline std::vector<int> query_gpu_nearby_logical_cores_from_nvidia_smi(int gpu_id) {
    const std::string cmd = "nvidia-smi topo -m 2>/dev/null";
    const std::string txt = slurp_command_output(cmd);
    if (txt.empty()) return {};

    std::stringstream ss(txt);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(ss, line)) {
        if (!trim(line).empty()) lines.push_back(line);
    }
    if (lines.size() < 2) return {};

    std::size_t affinity_pos = std::string::npos;
    std::size_t numa_pos = std::string::npos;
    {
        const std::string& header = lines[0];
        affinity_pos = header.find("CPU Affinity");
        numa_pos = header.find("NUMA Affinity");
        if (affinity_pos == std::string::npos) return {};
    }

    const std::string gpu_prefix = "GPU" + std::to_string(gpu_id);
    for (std::size_t li = 1; li < lines.size(); ++li) {
        const std::string row = lines[li];
        const std::string trow = trim(row);
        if (trow.rfind(gpu_prefix, 0) != 0) continue;

        std::string affinity_text;
        if (numa_pos != std::string::npos && numa_pos > affinity_pos) {
            affinity_text = trim(row.substr(affinity_pos, numa_pos - affinity_pos));
        } else {
            affinity_text = trim(row.substr(affinity_pos));
        }
        return expand_cpu_affinity_token(affinity_text);
    }

    return {};
}

static inline int query_gpu_nearby_numa_from_nvidia_smi(int gpu_id) {
    const std::string cmd = "nvidia-smi topo -m 2>/dev/null";
    const std::string txt = slurp_command_output(cmd);
    if (txt.empty()) return -1;

    std::stringstream ss(txt);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(ss, line)) {
        if (!trim(line).empty()) lines.push_back(line);
    }
    if (lines.size() < 2) return -1;

    const std::string& header = lines[0];
    const std::size_t numa_pos = header.find("NUMA Affinity");
    if (numa_pos == std::string::npos) return -1;

    const std::string gpu_prefix = "GPU" + std::to_string(gpu_id);
    for (std::size_t li = 1; li < lines.size(); ++li) {
        const std::string row = trim(lines[li]);
        if (row.rfind(gpu_prefix, 0) != 0) continue;

        std::string tail = trim(lines[li].substr(numa_pos));
        std::stringstream ts(tail);
        std::string first;
        ts >> first;
        try {
            return std::stoi(first);
        } catch (...) {
            return -1;
        }
    }

    return -1;
}

static inline std::vector<int> convert_physical_to_logical_cores(hwloc_topology_t topo, const std::vector<int>& physical_cores) {
    std::set<int> uniq;
    for (int pc : physical_cores) {
        const std::vector<int> lcs = physical_core_to_logical_cores(topo, pc);
        uniq.insert(lcs.begin(), lcs.end());
    }
    return std::vector<int>(uniq.begin(), uniq.end());
}

static inline bool bind_current_to_logical_cores_sched(const std::vector<int>& logical_cores, std::string* err = nullptr) {
    if (logical_cores.empty()) {
        if (err) *err = "empty logical core list";
        return false;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int c : logical_cores) {
        if (c >= 0) CPU_SET(c, &cpuset);
    }

    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        if (err) *err = std::strerror(errno);
        return false;
    }
    return true;
}

static inline bool bind_current_to_logical_cores_hwloc(hwloc_topology_t topo, const std::vector<int>& logical_cores, std::string* err = nullptr) {
    if (logical_cores.empty()) {
        if (err) *err = "empty logical core list";
        return false;
    }

    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    for (int c : logical_cores) {
        if (c >= 0) hwloc_bitmap_set(set, c);
    }

    const int rc = hwloc_set_cpubind(topo, set, HWLOC_CPUBIND_THREAD);
    hwloc_bitmap_free(set);
    if (rc != 0) {
        if (err) *err = std::strerror(errno);
        return false;
    }
    return true;
}

static inline std::vector<int> current_affinity_logical_cores() {
    std::vector<int> out;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) != 0) return out;

    const int nconf = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    for (int c = 0; c < nconf; ++c) {
        if (CPU_ISSET(c, &cpuset)) out.push_back(c);
    }
    return out;
}

static inline std::string join_int_vector(const std::vector<int>& v) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i + 1 < v.size()) oss << ",";
    }
    return oss.str();
}

static inline long long current_rss_bytes() {
    std::ifstream fin("/proc/self/status");
    std::string line;
    while (std::getline(fin, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::stringstream ss(line.substr(6));
            long long kb = 0;
            ss >> kb;
            return kb * 1024LL;
        }
    }
    return -1;
}

#ifdef __CUDACC__
static inline bool current_gpu_memory_bytes(long long& used_bytes, long long& free_bytes, long long& total_bytes) {
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    cudaError_t rc = cudaMemGetInfo(&free_b, &total_b);
    if (rc != cudaSuccess) {
        used_bytes = free_bytes = total_bytes = -1;
        return false;
    }
    free_bytes = static_cast<long long>(free_b);
    total_bytes = static_cast<long long>(total_b);
    used_bytes = total_bytes - free_bytes;
    return true;
}
#endif

static inline std::string human_bytes(long long bytes) {
    if (bytes < 0) return "N/A";
    static const char* suffix[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(bytes);
    int i = 0;
    while (v >= 1024.0 && i < 4) {
        v /= 1024.0;
        ++i;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision((i == 0) ? 0 : 2) << v << " " << suffix[i];
    return oss.str();
}

static inline BindingDecision decide_binding_for_rank(
    hwloc_topology_t topo,
    const RuntimeResourceOptions& opt,
    int rank,
    bool is_gpu_worker,
    int gpu_id,
    const std::vector<int>& reserved_gpu_host_logical_cores = {}
) {
    BindingDecision plan;

    if (opt.bind_mode == BindMode::Off) {
        plan.source = "binding disabled";
        return plan;
    }

    if (opt.bind_mode == BindMode::Manual) {
        if (is_gpu_worker) {
            if (rank < static_cast<int>(opt.gpu_host_cores.size())) {
                if (opt.core_kind == CoreKind::Logical) {
                    plan.selected_logical_cores = {opt.gpu_host_cores[rank]};
                    plan.selected_physical_cores = logical_to_physical_list(topo, plan.selected_logical_cores);
                } else {
                    plan.selected_physical_cores = {opt.gpu_host_cores[rank]};
                    plan.selected_logical_cores = convert_physical_to_logical_cores(topo, plan.selected_physical_cores);
                }
                plan.selected_numa = !plan.selected_logical_cores.empty()
                    ? find_numa_for_logical_core(topo, plan.selected_logical_cores.front()) : -1;
                plan.source = "manual gpu-host-cores";
            }
        } else {
            if (!opt.cpu_cores.empty()) {
                if (opt.core_kind == CoreKind::Logical) {
                    plan.selected_logical_cores = opt.cpu_cores;
                    plan.selected_physical_cores = logical_to_physical_list(topo, plan.selected_logical_cores);
                } else {
                    plan.selected_physical_cores = opt.cpu_cores;
                    plan.selected_logical_cores = convert_physical_to_logical_cores(topo, plan.selected_physical_cores);
                }
                if (!plan.selected_logical_cores.empty()) {
                    plan.selected_numa = find_numa_for_logical_core(topo, plan.selected_logical_cores.front());
                }
                plan.source = "manual cpu-cores";
            }
        }
        return plan;
    }

    if (is_gpu_worker) {
        std::vector<int> candidates = query_gpu_nearby_logical_cores_from_nvidia_smi(gpu_id);
        plan.selected_numa = query_gpu_nearby_numa_from_nvidia_smi(gpu_id);

        if (candidates.empty()) {
            const TopologySummary s = query_topology_summary(topo);
            candidates = s.logical_os_ids;
            plan.source = "auto fallback all logical cores";
        } else {
            plan.source = "auto nvidia-smi topo";
        }

        std::set<int> reserved(reserved_gpu_host_logical_cores.begin(), reserved_gpu_host_logical_cores.end());
        int chosen = -1;
        for (int c : candidates) {
            if (!reserved.count(c)) {
                chosen = c;
                break;
            }
        }
        if (chosen < 0 && !candidates.empty()) chosen = candidates.front();

        if (chosen >= 0) {
            plan.selected_logical_cores = {chosen};
            plan.selected_physical_cores = logical_to_physical_list(topo, plan.selected_logical_cores);
            if (plan.selected_numa < 0) {
                plan.selected_numa = find_numa_for_logical_core(topo, chosen);
            }
        }
        return plan;
    }

    TopologySummary s = query_topology_summary(topo);
    std::vector<int> cpu_pool;
    if (opt.cpu_numa >= 0) {
        cpu_pool = get_numa_logical_cores(topo, opt.cpu_numa);
        plan.selected_numa = opt.cpu_numa;
        plan.source = "auto numa restricted";
    } else {
        cpu_pool = get_interleaved_numa_logical_cores(topo);
        plan.source = "auto interleaved all logical cores";
    }

    std::set<int> banned(reserved_gpu_host_logical_cores.begin(), reserved_gpu_host_logical_cores.end());
    for (int c : cpu_pool) {
        if (!banned.count(c)) {
            plan.selected_logical_cores.push_back(c);
        }
    }
    plan.selected_physical_cores = logical_to_physical_list(topo, plan.selected_logical_cores);

    if (opt.cpu_threads > 0 && static_cast<int>(plan.selected_logical_cores.size()) > opt.cpu_threads) {
        plan.selected_logical_cores.resize(opt.cpu_threads);
        plan.selected_physical_cores = logical_to_physical_list(topo, plan.selected_logical_cores);
    }

    if (plan.selected_numa < 0 && !plan.selected_logical_cores.empty()) {
        plan.selected_numa = find_numa_for_logical_core(topo, plan.selected_logical_cores.front());
    }
    return plan;
}

static inline bool apply_binding_plan(hwloc_topology_t topo, const BindingDecision& plan, bool try_hwloc_first = false) {
    if (plan.selected_logical_cores.empty()) return false;

    std::string err;
    if (try_hwloc_first) {
        if (bind_current_to_logical_cores_hwloc(topo, plan.selected_logical_cores, &err)) {
            return true;
        }
    }
    return bind_current_to_logical_cores_sched(plan.selected_logical_cores, &err);
}

static inline void print_binding_report(
    std::ostream& os,
    int rank,
    const std::string& role,
    int gpu_id,
    const BindingDecision& plan,
    hwloc_topology_t topo
) {
    os << "[Rank " << rank << "] role=" << role;
    if (gpu_id >= 0) os << " gpu=" << gpu_id;
    os << "\n";

    os << "[Rank " << rank << "] binding source=" << plan.source << "\n";
    os << "[Rank " << rank << "] selected logical cores={" << join_int_vector(plan.selected_logical_cores) << "}\n";
    os << "[Rank " << rank << "] selected physical cores={" << join_int_vector(plan.selected_physical_cores) << "}\n";
    os << "[Rank " << rank << "] selected numa=" << plan.selected_numa << "\n";

    const std::vector<int> active = current_affinity_logical_cores();
    os << "[Rank " << rank << "] active affinity logical={" << join_int_vector(active) << "}\n";
    os << "[Rank " << rank << "] active affinity physical={" << join_int_vector(logical_to_physical_list(topo, active)) << "}\n";
}

static inline void print_monitor_snapshot(
    std::ostream& os,
    int rank,
    const std::string& role,
    int step,
    bool is_gpu_worker,
    double t_compute,
    double t_comm
) {
    os << "[Rank " << rank << "] step=" << step
       << " role=" << role
       << " compute=" << t_compute << " s"
       << " comm=" << t_comm << " s"
       << " rss=" << human_bytes(current_rss_bytes());

#ifdef __CUDACC__
    if (is_gpu_worker) {
        long long used_b = -1, free_b = -1, total_b = -1;
        if (current_gpu_memory_bytes(used_b, free_b, total_b)) {
            os << " gpu_mem_used=" << human_bytes(used_b)
               << " gpu_mem_free=" << human_bytes(free_b)
               << " gpu_mem_total=" << human_bytes(total_b);
        }
    }
#else
    (void)is_gpu_worker;
#endif

    os << "\n";
}

} // namespace rrc

#endif
