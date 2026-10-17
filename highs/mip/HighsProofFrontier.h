#ifndef HIGHS_PROOF_FRONTIER_H_
#define HIGHS_PROOF_FRONTIER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Single-owner coverage contract for a minimization tree. This is not a dual
// certificate checker and is not connected to live pruning. Callers may supply
// only independently established node bounds. Transfer never removes coverage.
class HighsProofFrontier {
 public:
  struct Generation {
    std::string model;
    std::string mapping;
    uint64_t solve;
    uint64_t stage;
    uint64_t restart;
    bool operator==(const Generation& other) const {
      return model == other.model && mapping == other.mapping &&
             solve == other.solve && stage == other.stage &&
             restart == other.restart;
    }
  };
  enum class State { Ready, Running, Suspended, InTransit };
  struct Token {
    Generation generation;
    uint64_t id;
    uint64_t version;
    uint64_t worker;
  };
  struct Child {
    uint64_t id;
    double bound;
  };
  struct Entry {
    uint64_t version;
    State state;
    double bound;
    uint64_t worker;
  };

  explicit HighsProofFrontier(const Generation& generation)
      : generation_(generation) {}
  HighsProofFrontier(const HighsProofFrontier&) = delete;
  HighsProofFrontier& operator=(const HighsProofFrontier&) = delete;

  bool initialize(uint64_t id, double bound) {
    if (initialized_ || generation_.model.empty() ||
        generation_.mapping.empty() || !validBound(bound))
      return false;
    std::map<uint64_t, Entry> nodes;
    std::set<uint64_t> seen;
    nodes.emplace(id, Entry{0, State::Ready, bound, 0});
    seen.insert(id);
    nodes_.swap(nodes);
    seen_.swap(seen);
    initialized_ = true;
    return true;
  }
  bool token(uint64_t id, Token& out) const {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return false;
    out = {generation_, id, it->second.version, it->second.worker};
    return true;
  }
  bool transition(Token in, State from, State to, Token& out,
                  uint64_t receiving_worker = 0) {
    auto it = find(in);
    if (it == nodes_.end() || it->second.state != from ||
        it->second.version == std::numeric_limits<uint64_t>::max())
      return false;
    const bool permitted =
        ((from == State::Ready || from == State::Suspended) &&
         to == State::Running) ||
        (from == State::Running &&
         (to == State::Suspended || to == State::InTransit)) ||
        (from == State::InTransit && to == State::Ready);
    if (!permitted) return false;
    const uint64_t worker =
        to == State::Running ? receiving_worker : it->second.worker;
    // Copy receipt storage before changing ownership: even allocation failure
    // while copying generation strings must leave the old lease reachable.
    Token next{generation_, in.id, it->second.version + 1, worker};
    out = std::move(next);
    ++it->second.version;
    it->second.state = to;
    it->second.worker = worker;
    return true;
  }
  bool strengthen(Token in, double bound) {
    auto it = find(in);
    if (it == nodes_.end() || it->second.state != State::Running ||
        !validBound(bound) || bound < it->second.bound)
      return false;
    it->second.bound = bound;
    return true;
  }
  bool replaceWithChildren(Token parent, const std::vector<Child>& children) {
    auto it = find(parent);
    if (it == nodes_.end() || it->second.state != State::Running ||
        children.empty())
      return false;
    std::set<uint64_t> ids;
    for (const auto& child : children)
      if (seen_.count(child.id) || !ids.insert(child.id).second ||
          !validBound(child.bound) || child.bound < it->second.bound)
        return false;
    // Transactional staging: allocation failure leaves the parent and all
    // coverage intact. A later live implementation needs a bounded delta
    // commit.
    auto nodes = nodes_;
    auto seen = seen_;
    for (const auto& child : children) {
      nodes.emplace(child.id, Entry{0, State::Ready, child.bound, 0});
      seen.insert(child.id);
    }
    nodes.erase(parent.id);
    nodes_.swap(nodes);
    seen_.swap(seen);
    return true;
  }
  bool closeInfeasible(Token in) {
    auto it = find(in);
    if (it == nodes_.end() || it->second.state != State::Running) return false;
    nodes_.erase(it);
    return true;
  }
  bool closeWithBound(Token in, double certified_bound) {
    auto it = find(in);
    if (it == nodes_.end() || it->second.state != State::Running ||
        !validBound(certified_bound) || certified_bound < it->second.bound)
      return false;
    // Includes exact solutions and tolerance/cutoff closures. Never replace an
    // actual closure bound with the current incumbent or requested gap target.
    closed_bound_ = std::min(closed_bound_, certified_bound);
    nodes_.erase(it);
    return true;
  }
  double lowerBound() const {
    if (!initialized_) return -std::numeric_limits<double>::infinity();
    double bound = closed_bound_;
    for (const auto& node : nodes_) bound = std::min(bound, node.second.bound);
    return bound;
  }
  bool reconciled() const { return initialized_ && nodes_.empty(); }
  size_t size() const { return nodes_.size(); }
  size_t count(State state) const {
    size_t n = 0;
    for (const auto& node : nodes_) n += node.second.state == state;
    return n;
  }

 private:
  static bool validBound(double x) {
    return !std::isnan(x) && x != std::numeric_limits<double>::infinity();
  }
  std::map<uint64_t, Entry>::iterator find(Token token) {
    if (!(token.generation == generation_)) return nodes_.end();
    auto it = nodes_.find(token.id);
    return it != nodes_.end() && it->second.version == token.version &&
                   it->second.worker == token.worker
               ? it
               : nodes_.end();
  }
  bool initialized_ = false;
  const Generation generation_;
  double closed_bound_ = std::numeric_limits<double>::infinity();
  std::map<uint64_t, Entry> nodes_;
  std::set<uint64_t> seen_;
};
#endif
