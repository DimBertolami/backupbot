#!/bin/bash

# Ensure we're in the correct directory
cd /opt/lampp/htdocs/bot/frontend || exit

# Install dependencies if needed
npm install -g jest-coverage-reporter

# Run tests with coverage
npm test -- --coverage

# Generate coverage report
jest-coverage-reporter

# Calculate coverage percentage
COVERAGE=$(cat coverage/coverage-final.json | jq -r '.total.statements.pct')

# Check if coverage has decreased
PREV_COVERAGE=$(cat coverage/previous_coverage.txt 2>/dev/null || echo "0")

# Save current coverage for next run
echo "$COVERAGE" > coverage/previous_coverage.txt

# Check for regression
if (( $(echo "$COVERAGE < $PREV_COVERAGE" | bc -l) )); then
    echo "WARNING: Test coverage has decreased from $PREV_COVERAGE% to $COVERAGE%"
    echo "Please review changes that might have reduced test coverage"
fi

# Print coverage summary
echo "Test Coverage Summary:"
echo "Statements: ${COVERAGE}%"
echo "Branches: $(cat coverage/coverage-final.json | jq -r '.total.branches.pct')%"
echo "Functions: $(cat coverage/coverage-final.json | jq -r '.total.functions.pct')%"
echo "Lines: $(cat coverage/coverage-final.json | jq -r '.total.lines.pct')%"

# Generate HTML report
npx jest --coverage --coverage-reporters="html"

# Check for failing tests
TEST_STATUS=$?

if [ $TEST_STATUS -ne 0 ]; then
    echo "ERROR: Some tests failed"
    exit 1
fi

# Generate test coverage report
echo "Generating test coverage report..."
npx jest-coverage-reporter --reporter="html" --reporter="lcov" --reporter="text"

# Create summary file
echo "Test Coverage Summary" > coverage/coverage_summary.txt
echo "====================" >> coverage/coverage_summary.txt
echo "" >> coverage/coverage_summary.txt
echo "Statements: ${COVERAGE}%" >> coverage/coverage_summary.txt
echo "Branches: $(cat coverage/coverage-final.json | jq -r '.total.branches.pct')%" >> coverage/coverage_summary.txt
echo "Functions: $(cat coverage/coverage-final.json | jq -r '.total.functions.pct')%" >> coverage/coverage_summary.txt
echo "Lines: $(cat coverage/coverage-final.json | jq -r '.total.lines.pct')%" >> coverage/coverage_summary.txt

echo "Test coverage analysis complete."
exit 0
