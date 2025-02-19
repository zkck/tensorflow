/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/hlo_live_range.h"

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"

namespace xla {
/*static*/
StatusOr<std::unique_ptr<HloLiveRange>> HloLiveRange::Run(
    const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
    const HloComputation* computation, bool module_scoped_analysis) {
  std::unique_ptr<HloLiveRange> hlo_live_range(
      new HloLiveRange(schedule, alias_analysis, module_scoped_analysis));
  hlo_live_range->FlattenSchedule(*computation);
  hlo_live_range->CalculateBufferStartEndMap();
  hlo_live_range->NormalizeAliasedBuffers();
  return std::move(hlo_live_range);
}

void HloLiveRange::NormalizeAliasedBuffers() {
  for (const HloBuffer& hlo_buffer : alias_analysis_.buffers()) {
    std::vector<TimeBound*> aliased_live_ranges;
    aliased_live_ranges.reserve(hlo_buffer.values().size());
    for (const HloValue* hlo_value : hlo_buffer.values()) {
      auto it = buffer_live_ranges_.find(hlo_value);
      if (it != buffer_live_ranges_.end()) {
        aliased_live_ranges.push_back(&it->second);
      }
    }
    absl::c_sort(aliased_live_ranges,
                 [](const TimeBound* a, const TimeBound* b) {
                   return std::forward_as_tuple(a->start, a->end) <
                          std::forward_as_tuple(b->start, b->end);
                 });

    for (int64_t i = 0; i + 1 < aliased_live_ranges.size(); ++i) {
      TimeBound& live_range1 = *aliased_live_ranges[i];
      TimeBound& live_range2 = *aliased_live_ranges[i + 1];
      if (live_range1.start == live_range2.start) {
        // If value1 has the same start time as value2, make value1 disappear
        // by setting the end time same as start time:
        //
        // Before:
        // +----+           value1
        // +----------+     value2
        //
        // After:
        // +                value1
        // +----------+     value2
        //
        // Note that only when heap simulator runs before copy insertion can
        // this happen where one instruction defines multiple aliased buffers
        // -- This is illegal to execute and can be fixed by copy insertion
        // later.
        // FIXME(cjfj): This code doesn't match the behaviour described above.
        live_range1.end = live_range2.end;
        continue;
      }

      if (live_range1.end < live_range2.start) {
        continue;
      }

      live_range2.end = std::max(live_range1.end, live_range2.end);
      live_range1.end = live_range2.start - 1;
    }
  }
}

// FlattenSchedule walks through the computation and tracks down the ordinal
// number of each instruction in the schedule.
void HloLiveRange::FlattenSchedule(const HloComputation& computation) {
  auto it = schedule_.sequences().find(computation.unique_id());
  if (it == schedule_.sequences().end()) {
    total_order_scheduled_ = false;
    return;
  }

  // Check if we've already processed this computation.
  if (computation_span_times_.contains(&computation)) return;

  int64_t start_time = flattened_instruction_sequence_.size();

  const HloInstructionSequence& instruction_sequence = it->second;
  for (HloInstruction* instruction : instruction_sequence.instructions()) {
    if (module_scoped_analysis_) {
      // Recurse into sub computations if running with module scoped analysis
      // mode.
      if (instruction->opcode() == HloOpcode::kCall ||
          instruction->opcode() == HloOpcode::kConditional) {
        for (const HloComputation* called_computation :
             instruction->called_computations()) {
          FlattenSchedule(*called_computation);
        }
      } else if (instruction->opcode() == HloOpcode::kWhile) {
        FlattenSchedule(*instruction->while_condition());
        FlattenSchedule(*instruction->while_body());
      }
    }

    int64_t time = flattened_instruction_sequence_.size();
    CHECK(instruction_schedule_.insert({instruction, time}).second);
    flattened_instruction_sequence_.push_back(instruction);
  }

  int64_t end_time = flattened_instruction_sequence_.size();
  computation_span_times_[&computation] = {start_time, end_time};
}

void HloLiveRange::CalculateBufferStartEndMap() {
  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    auto it = instruction_schedule_.find(value->defining_instruction());
    // Ignore buffers that are not defined.
    if (it == instruction_schedule_.end()) continue;

    int64_t buffer_start_time = it->second;

    // Parameters are defined at the beginning of the computation. This prevents
    // any instruction that's scheduled before the parameter clobbers the
    // parameter's buffer.
    if (value->instruction()->opcode() == HloOpcode::kParameter) {
      const HloComputation* computation = value->instruction()->parent();
      auto it = computation_span_times_.find(computation);
      if (it != computation_span_times_.end()) {
        buffer_start_time = std::min(buffer_start_time, it->second.start);
      }
    }

    int64_t buffer_end_time = buffer_start_time;

    for (const HloUse& use : value->uses()) {
      const HloInstruction* used = use.instruction;
      // As an optimization, we deem a while's init value's live range ends as
      // soon as the loop body starts. This optimization is only applicable in
      // module scoped mode.
      if (module_scoped_analysis_ && used->opcode() == HloOpcode::kWhile) {
        // The current live range is at the end of the while, move it to the
        // beginning of the body.
        used = used->while_body()->parameter_instruction(0);
        VLOG(1) << "Moved value " << value->ToShortString()
                << " to while param: " << used->ToString();
      }

      // It's possible that we didn't track the instruction `used`. This happens
      // when we do computation scope (versus module scope) heap simulation and
      // the used instruction is outside of the computation being simulated.
      auto it = instruction_schedule_.find(used);
      if (it != instruction_schedule_.end()) {
        buffer_end_time = std::max(buffer_end_time, it->second);
      }
    }

    HloPosition end_position;
    int64_t max_end_time = 0;
    for (const HloPosition& position : value->positions()) {
      if (instruction_schedule_[position.instruction] >= max_end_time) {
        max_end_time = instruction_schedule_[value->instruction()];
        end_position = position;
      }
      const HloComputation* position_comp = position.instruction->parent();
      // If this instruction lives out, the live range of the instruction
      // should be extended to the end of the computation.
      if (position.instruction == position_comp->root_instruction()) {
        auto it = computation_span_times_.find(position_comp);
        if (it != computation_span_times_.end()) {
          if (buffer_end_time < it->second.end) {
            buffer_end_time = it->second.end;
            end_position = position;
          }
        }
      }
    }

    const HloModule* module = value->instruction()->parent()->parent();

    // Readonly entry parameters (parameters that don't alias) live across whole
    // computation.
    if (value->instruction()->opcode() == HloOpcode::kParameter &&
        value->instruction()->parent() == module->entry_computation() &&
        !module->input_output_alias_config().ParameterHasAlias(
            value->instruction()->parameter_number(), value->index())) {
      buffer_end_time = schedule_end_time();
    }

    CHECK(buffer_start_time <= buffer_end_time)
        << buffer_start_time << ", " << buffer_end_time << ": "
        << value->instruction()->ToString();

    bool was_inserted =
        buffer_live_ranges_
            .insert({value, {buffer_start_time, buffer_end_time, end_position}})
            .second;
    CHECK(was_inserted) << "Value live range already calculated";
  }
}

int64_t HloLiveRange::ComputePeakMemoryMoment() const {
  std::vector<std::tuple<int64_t /*time*/, bool /*is_end*/, const HloValue*>>
      events;
  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    auto it = buffer_live_ranges_.find(value);
    if (it != buffer_live_ranges_.end()) {
      events.emplace_back(it->second.start, false, value);
      events.emplace_back(it->second.end + 1, true, value);
    }
  }
  std::sort(events.begin(), events.end());

  int64_t memory_usage = 0;
  int64_t peak_usage = 0;
  absl::optional<int64_t> peak_time;
  for (const auto& event : events) {
    int64_t time;
    bool is_end;
    const HloValue* value;
    std::tie(time, is_end, value) = event;
    auto buffer_size = ShapeUtil::ByteSizeOf(value->instruction()->shape(), 8);
    if (is_end) {
      memory_usage -= buffer_size;
    } else {
      memory_usage += buffer_size;
    }
    if (peak_usage < memory_usage) {
      peak_usage = memory_usage;
      peak_time = time;
    }
  }
  return peak_time.value_or(0);
}

std::string HloLiveRange::ToString() const {
  std::string output;
  absl::StrAppendFormat(&output, "HloLiveRange (max %d):\n",
                        schedule_end_time());
  absl::StrAppendFormat(&output, "  InstructionSequence:\n");
  auto& instructions = flattened_instruction_sequence().instructions();
  for (int64_t i = 0; i < instructions.size(); ++i) {
    absl::StrAppendFormat(&output, "    %d:%s\n", i, instructions[i]->name());
  }

  absl::StrAppendFormat(&output, "  BufferLiveRange:\n");

  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    auto it = buffer_live_ranges_.find(value);
    if (it != buffer_live_ranges_.end()) {
      absl::StrAppendFormat(
          &output, "    %s%s:%d-%d\n", value->instruction()->name(),
          value->index().ToString(), it->second.start, it->second.end);
    }
  }

  int64_t peak_moment = ComputePeakMemoryMoment();

  absl::StrAppendFormat(&output, "  Live ranges at %lld (peak):\n",
                        peak_moment);
  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    auto it = buffer_live_ranges_.find(value);
    if (it != buffer_live_ranges_.end()) {
      if (it->second.start <= peak_moment && peak_moment <= it->second.end) {
        int64_t bytes = ShapeUtil::ByteSizeOf(value->instruction()->shape(), 8);
        absl::StrAppendFormat(&output, "    %s: %lld bytes\n",
                              value->instruction()->name(), bytes);
      }
    }
  }

  return output;
}

}  // namespace xla
