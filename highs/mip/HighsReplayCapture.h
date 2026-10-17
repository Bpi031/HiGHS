#ifndef HIGHS_REPLAY_CAPTURE_H_
#define HIGHS_REPLAY_CAPTURE_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include "Highs.h"

struct HighsReplayContext {
  const char* method = nullptr;
  const char* parent_phase = nullptr;
  uint64_t decision = 0;
  HighsInt worker = -1;
  HighsInt restart = -1;
  double parent_time = -1;
  HighsInt lp_solution_status = kSolutionStatusNone;
  const std::vector<double>* relaxation = nullptr;
  const std::vector<double>* base_lower = nullptr;
  const std::vector<double>* base_upper = nullptr;
  const std::vector<double>* checked_seed = nullptr;
  bool original_space = false;
  const HighsOptions* options = nullptr;
  int64_t helper_lp_limit = -1;
};

// Explicit diagnostic opt-in. Cold replay preserves the matrix, local bounds,
// integrality and offset, but not propagation/clique/pseudocost caches.
class HighsReplayCapture {
 public:
  static bool enabled(const char* kind, const char* method = nullptr) {
    const auto& sink = instance();
    return !sink.directory_.empty() && kind && selected(sink.filter_, kind, method);
  }
  static bool selected(const std::string& filter, const std::string& kind,
                       const char* method = nullptr) {
    if (kind != "node_lp" && kind != "sub_mip") return false;
    if (filter.empty() || filter == "all") return true;
    if (filter == "node_lp") return kind == "node_lp";
    if (kind != "sub_mip" || !method) return false;
    for (const char* known : {"rens", "rins", "root_reduced_cost", "local_branching"})
      if (filter == known) return filter == method;
    return false;
  }

  static size_t captureEstimate(const HighsLp& lp, bool context) {
    constexpr size_t cap = 64 * 1024 * 1024;
    if (lp.num_col_ < 0 || lp.num_row_ < 0) return cap + 1;
    size_t size = 4096;
    auto add = [&](size_t count, size_t weight) {
      if (count > (cap - size) / weight) return false;
      size += count * weight;
      return true;
    };
    if (!add(lp.a_matrix_.value_.size(), 128) ||
        !add(static_cast<size_t>(lp.num_col_), context ? 768 : 512) ||
        !add(static_cast<size_t>(lp.num_row_), 512)) return cap + 1;
    for (const auto& names : {&lp.col_names_, &lp.row_names_})
      for (const auto& name : *names)
        if (!add(name.size(), 4)) return cap + 1;
    return size;
  }

  static bool validContext(const HighsLp& lp, const HighsReplayContext& context) {
    const std::string method = context.method ? context.method : "";
    if (!context.method || !context.parent_phase ||
        (method != "rens" && method != "rins" &&
         method != "root_reduced_cost" && method != "local_branching") ||
        (std::string(context.parent_phase) != "tree" &&
         std::string(context.parent_phase) != "root")) return false;
    if (!context.relaxation && !context.original_space) return true;
    if (lp.num_col_ < 0 || lp.num_row_ < 0) return false;
    const size_t n = static_cast<size_t>(lp.num_col_);
    if (context.decision == 0 || context.worker < 0 || context.restart < 0 ||
        !std::isfinite(context.parent_time) || context.parent_time < 0 ||
        context.helper_lp_limit < 0 ||
        context.lp_solution_status < kSolutionStatusNone ||
        context.lp_solution_status > kSolutionStatusFeasible ||
        !context.options || !context.base_lower || !context.base_upper ||
        (context.original_space && (!context.checked_seed || context.checked_seed->size() != n ||
                                   context.relaxation)) ||
        (!lp.integrality_.empty() && lp.integrality_.size() != n) ||
        (!context.relaxation && !context.original_space) ||
        (context.relaxation && context.relaxation->size() != n) || context.base_lower->size() != n ||
        context.base_upper->size() != n ||
        (context.checked_seed && !context.checked_seed->empty() &&
         context.checked_seed->size() != n)) return false;
    const auto& options = *context.options;
    if ((options.parallel != "on" && options.parallel != "off" &&
         options.parallel != "choose") ||
        (options.presolve != "on" && options.presolve != "off" &&
         options.presolve != "choose") ||
        std::isnan(options.time_limit) || options.time_limit <= 0 ||
        std::isnan(options.objective_bound) ||
        !std::isfinite(options.mip_feasibility_tolerance) ||
        options.mip_feasibility_tolerance <= 0 ||
        !std::isfinite(options.mip_rel_gap) || options.mip_rel_gap < 0 ||
        !std::isfinite(options.mip_abs_gap) || options.mip_abs_gap < 0 ||
        !std::isfinite(options.mip_heuristic_effort) ||
        options.mip_heuristic_effort < 0 || options.mip_heuristic_effort > 1)
      return false;
    for (size_t col = 0; col < n; ++col) {
      if ((context.relaxation && !std::isfinite((*context.relaxation)[col])) ||
          std::isnan((*context.base_lower)[col]) ||
          std::isnan((*context.base_upper)[col]) ||
          (*context.base_lower)[col] > (*context.base_upper)[col] ||
          (context.checked_seed && !context.checked_seed->empty() &&
           !std::isfinite((*context.checked_seed)[col]))) return false;
    }
    return true;
  }

  static HighsLp modelForReplay(const HighsLp& lp, bool node) {
    HighsLp captured = lp;
    if (node) captured.integrality_.clear();
    // Cut rows can be unnamed. Canonical names avoid writer warnings without
    // accepting numerical model-assessment warnings or changing live state.
    captured.col_names_.resize(lp.num_col_);
    captured.row_names_.resize(lp.num_row_);
    for (HighsInt col = 0; col < lp.num_col_; ++col)
      captured.col_names_[col] = "c" + std::to_string(col);
    for (HighsInt row = 0; row < lp.num_row_; ++row)
      captured.row_names_[row] = "r" + std::to_string(row);
    captured.objective_name_ = "OBJ";
    captured.col_hash_.name2index.clear();
    captured.row_hash_.name2index.clear();
    return captured;
  }

  static bool capture(const HighsLp& lp, const char* kind, HighsInt depth,
                      const HighsBasis* basis = nullptr,
                      const HighsReplayContext* context = nullptr) {
    auto& sink = instance();
    if (sink.directory_.empty() || !kind ||
        !selected(sink.filter_, kind, context ? context->method : nullptr)) return false;
    if (context && !validContext(lp, *context)) return false;
    std::lock_guard<std::mutex> lock(sink.mutex_);
    const bool node = std::string(kind) == "node_lp";
    size_t& count = node ? sink.nodes_ : sink.submips_;
    if (count >= 6) return false;
    const bool construction = context && (context->relaxation || context->original_space);
    const size_t estimate = captureEstimate(lp, construction);
    if (estimate > sink.remaining_) return false;
    const std::string base = sink.directory_ + "/" + kind + "-" +
                             std::to_string(count);
    FILE* reservation = std::fopen((base + ".json").c_str(), "wx");
    if (!reservation) return false;
    std::fclose(reservation);
    FILE* model_reservation = std::fopen((base + ".mps").c_str(), "wx");
    if (!model_reservation) {
      std::remove((base + ".json").c_str());
      return false;
    }
    std::fclose(model_reservation);
    ++count;
    Highs replay;
    replay.setOptionValue("output_flag", false);
    HighsLp captured = modelForReplay(lp, node);
    if (replay.passModel(captured) != HighsStatus::kOk ||
        replay.writeModel(base + ".mps") != HighsStatus::kOk) {
      std::remove((base + ".mps").c_str());
      std::remove((base + ".json").c_str());
      return false;
    }
    std::ifstream bytes(base + ".mps", std::ios::binary | std::ios::ate);
    const auto length = bytes.tellg();
    const size_t meta_reserve = 4096 + (construction ? size_t{256} : size_t{8}) *
        static_cast<size_t>(lp.num_col_) + size_t{8} * static_cast<size_t>(lp.num_row_);
    if (length < 0 || static_cast<size_t>(length) + meta_reserve > sink.remaining_) {
      bytes.close();
      std::remove((base + ".mps").c_str());
      std::remove((base + ".json").c_str());
      return false;
    }
    bytes.close();
    const bool saved_basis = (node || construction) && basis && basis->valid &&
        basis->col_status.size() == static_cast<size_t>(lp.num_col_) &&
        basis->row_status.size() == static_cast<size_t>(lp.num_row_);
    std::ofstream meta(base + ".json", std::ios::trunc);
    meta.precision(17);
    meta << "{\"schema\":\"rost/highs-local-replay/v1\",\"kind\":\""
         << kind << "\",\"scope\":\"restricted_model_only\","
         << "\"cold_replay_omits_solver_caches\":true,"
         << "\"names\":\"canonical_index_order\",\"depth\":" << depth
         << ",\"rows\":" << lp.num_row_ << ",\"columns\":" << lp.num_col_
         << ",\"nonzeros\":" << lp.a_matrix_.numNz()
         << ",\"objective_offset\":" << lp.offset_
         << ",\"objective_sense\":" << static_cast<int>(lp.sense_)
         << ",\"basis_available\":" << (saved_basis ? "true" : "false")
         << ",\"contributes_global_bound\":false";
    if (context) {
      meta << ",\"method\":\"" << context->method << "\",\"parent_phase\":\""
           << context->parent_phase << '"';
    }
    if (construction) {
      writeContext(meta, *context);
      meta << ",\"integrality\":[";
      for (HighsInt col = 0; col < lp.num_col_; ++col) {
        if (col) meta << ',';
        meta << (lp.integrality_.empty() ? 0 : static_cast<int>(lp.integrality_[col]));
      }
      meta << ']';
    }
    if (saved_basis) {
      meta << ",\"basis_col_status\":[";
      for (size_t i = 0; i < basis->col_status.size(); ++i) {
        if (i) meta << ',';
        meta << static_cast<int>(basis->col_status[i]);
      }
      meta << "],\"basis_row_status\":[";
      for (size_t i = 0; i < basis->row_status.size(); ++i) {
        if (i) meta << ',';
        meta << static_cast<int>(basis->row_status[i]);
      }
      meta << ']';
    }
    meta << "}\n";
    meta.close();
    std::ifstream metadata(base + ".json", std::ios::binary | std::ios::ate);
    const auto meta_length = metadata.tellg();
    if (!meta || meta_length < 0 ||
        static_cast<size_t>(meta_length) > meta_reserve) {
      metadata.close();
      std::remove((base + ".mps").c_str());
      std::remove((base + ".json").c_str());
      return false;
    }
    sink.remaining_ -= static_cast<size_t>(length) + static_cast<size_t>(meta_length);
    return true;
  }

 private:
  static HighsReplayCapture& instance() {
    static HighsReplayCapture sink;
    return sink;
  }
  static void number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
    else if (std::isinf(value)) out << (value > 0 ? "\"inf\"" : "\"-inf\"");
    else out << "null";
  }
  static void vector(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t j = 0; j < values.size(); ++j) {
      if (j) out << ',';
      number(out, values[j]);
    }
    out << ']';
  }
  static void writeContext(std::ostream& out, const HighsReplayContext& context) {
    const auto& options = *context.options;
    out << ",\"construction_context\":{\"schema\":\"rost/highs-construction-context/v1\","
        << "\"decision\":" << context.decision << ",\"worker\":" << context.worker
        << ",\"restart\":" << context.restart << ",\"parent_time\":" << context.parent_time
        << ",\"lp_solution_status\":" << context.lp_solution_status
        << ",\"coordinates\":\"" << (context.original_space ? "original" : "reduced")
        << "\",\"relaxation\":";
    if (context.relaxation) vector(out, *context.relaxation);
    else out << "null";
    out << ",\"base_lower\":";
    vector(out, *context.base_lower);
    out << ",\"base_upper\":";
    vector(out, *context.base_upper);
    out << ",\"checked_seed\":";
    if (context.checked_seed) vector(out, *context.checked_seed);
    else out << "[]";
    out << ",\"helper_lp_limit\":" << context.helper_lp_limit
        << ",\"options_at_capture\":{\"threads\":" << options.threads
        << ",\"parallel\":\"" << options.parallel << "\",\"presolve\":\""
        << options.presolve << "\",\"random_seed\":" << options.random_seed
        << ",\"simplex_iteration_limit\":" << options.simplex_iteration_limit
        << ",\"mip_max_nodes\":" << options.mip_max_nodes
        << ",\"mip_max_leaves\":" << options.mip_max_leaves
        << ",\"mip_max_stall_nodes\":" << options.mip_max_stall_nodes
        << ",\"mip_pscost_minreliable\":" << options.mip_pscost_minreliable
        << ",\"time_limit\":";
    number(out, options.time_limit);
    out << ",\"objective_bound\":";
    number(out, options.objective_bound);
    out << ",\"mip_feasibility_tolerance\":" << options.mip_feasibility_tolerance
        << ",\"mip_rel_gap\":" << options.mip_rel_gap
        << ",\"mip_abs_gap\":" << options.mip_abs_gap
        << ",\"mip_heuristic_effort\":" << options.mip_heuristic_effort
        << "},\"missing_state\":[\"factors_pricing\",\"propagation_clique_pseudocost_caches\","
        << "\"original_postsolve_mapping\",\"parent_checkpoint_schedule\"]}";
  }

  HighsReplayCapture() {
    const char* value = std::getenv("ROST_HIGHS_REPLAY_DIR");
    if (value && value[0] == '/') directory_ = value;
    const char* filter = std::getenv("ROST_HIGHS_REPLAY_METHOD");
    if (filter) filter_ = filter;
  }
  std::string directory_;
  std::string filter_;
  std::mutex mutex_;
  size_t nodes_ = 0;
  size_t submips_ = 0;
  size_t remaining_ = 64 * 1024 * 1024;
};
#endif
