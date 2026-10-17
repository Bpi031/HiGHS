# Replay capture is process-wide; each invocation needs a new empty directory.
if(NOT IS_ABSOLUTE "${TEST_BINARY}" OR NOT EXISTS "${TEST_BINARY}"
   OR NOT IS_ABSOLUTE "${CAPTURE_ROOT}")
  message(FATAL_ERROR "Provide an absolute test binary and capture root")
endif()
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef run_id)
set(capture_directory "${CAPTURE_ROOT}/run-${run_id}")
if(EXISTS "${capture_directory}")
  message(FATAL_ERROR "Refusing to reuse replay capture evidence")
endif()
file(MAKE_DIRECTORY "${capture_directory}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "ROST_HIGHS_REPLAY_DIR=${capture_directory}" "ROST_HIGHS_REPLAY_METHOD=rens"
    "${TEST_BINARY}" "[construction_capture_io]"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 30)
message("${output}${errors}")
if(NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "Replay capture fixture failed: ${result}; evidence: ${capture_directory}")
endif()
if("${output}${errors}" MATCHES "No test cases matched"
   OR NOT "${output}${errors}" MATCHES "All tests passed \\([1-9][0-9]* assertions in 1 test case\\)")
  message(FATAL_ERROR "Missing executed replay capture assertions")
endif()
