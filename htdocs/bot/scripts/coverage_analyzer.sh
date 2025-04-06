#!/bin/bash

# Ensure we're in the frontend directory
cd /opt/lampp/htdocs/bot/frontend || exit

# Run tests with coverage
npm run test:coverage

# Generate detailed coverage report
npm run coverage:report

# Check coverage thresholds
npm run coverage:check

# Get coverage statistics
COVERAGE_STATS=$(cat coverage/coverage-final.json | jq -r '.total')

# Print coverage summary
echo "\nTest Coverage Summary:"
echo "===================="
echo "Statements: ${COVERAGE_STATS.statements.pct}%"
echo "Branches: ${COVERAGE_STATS.branches.pct}%"
echo "Functions: ${COVERAGE_STATS.functions.pct}%"
echo "Lines: ${COVERAGE_STATS.lines.pct}%"

# Generate coverage report for each component
echo "\nGenerating component-specific coverage reports..."
for COMPONENT in $(find src/components -type d -maxdepth 1 -mindepth 1); do
  COMPONENT_NAME=$(basename "$COMPONENT")
  COVERAGE=$(cat coverage/coverage-final.json | \
    jq -r --arg component "$COMPONENT_NAME" '.coverage["src/components/$component/$component.tsx"] | .statements.pct')
  echo "${COMPONENT_NAME}: ${COVERAGE}%"
done

# Generate HTML report and open it
npm run coverage:open

# Save coverage report for comparison
mkdir -p coverage/history
cp coverage/coverage-final.json coverage/history/coverage-$(date +%Y-%m-%d_%H-%M-%S).json

# Generate trend report
echo "\nGenerating coverage trend report..."
LATEST=$(ls -t coverage/history/*.json | head -n 1)
PREVIOUS=$(ls -t coverage/history/*.json | head -n 2 | tail -n 1)

if [ -f "$PREVIOUS" ]; then
  echo "\nCoverage Trend:"
  echo "==============="
  
  for METRIC in statements branches functions lines; do
    LATEST_VAL=$(cat "$LATEST" | jq -r ".total.${METRIC}.pct")
    PREV_VAL=$(cat "$PREVIOUS" | jq -r ".total.${METRIC}.pct")
    
    if (( $(echo "$LATEST_VAL < $PREV_VAL" | bc -l) )); then
      echo "${METRIC}: $LATEST_VAL% (↓ Decrease from $PREV_VAL%)"
    elif (( $(echo "$LATEST_VAL > $PREV_VAL" | bc -l) )); then
      echo "${METRIC}: $LATEST_VAL% (↑ Increase from $PREV_VAL%)"
    else
      echo "${METRIC}: $LATEST_VAL% (No change)"
    fi
  done
fi
