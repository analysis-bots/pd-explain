A test module for pd-explain using pytest.\
To run the tests, simply run `pytest` in the root directory of the project.\
\
These tests are meant to make sure that pd-explain produces output that is consistent with pandas' output.\
This is done to ensure that pd-explain does not modify the data in any way, and only provides a 
convenient way to get explanations for pandas objects.\
\
Test names are split into the following parts:
- `test_` - All test functions start with this prefix.\
- `<function_name>` - The name of the function being tested.\
- `<additional_info>` - Additional info, such as specific parameters used in the test. Not all tests have this.\
- `<expected_result>` - The expected result of the test. should_work means that the test should pass if the function produces exactly the expected output.
should_fail means that the test should pass if the function fails to produce output, raising an expected exception instead.