> Interesting the score went down from 0.765 to 0.725

⏺ Interesting indeed! This is actually a classic machine learning problem: overfitting. Adding more features made your model too complex and it memorized the training data instead of learning generalizable patterns.

  What's Happening: Overfitting

  Your decision tree with default parameters has no depth limit - it keeps splitting until it perfectly (or nearly perfectly) classifies the training data.

  With just Sex (simple model):

  - Creates 1-2 splits: "If female → survived, if male → died"
  - Simple rule generalizes well to test data
  - Score: 0.765

  With Sex, Age, and Pclass (complex model):

  - Creates 50+ splits with complex rules like "If female AND age < 23.5 AND Pclass = 3 AND age > 18.2..."
  - Memorizes noise/outliers in training data
  - Performs poorly on test data because those specific patterns don't repeat
  - Score: 0.725 ⬇️

  How to Fix It: Add Constraints

  The solution is to limit the tree's complexity so it learns general patterns, not specific training data quirks:

  from sklearn import tree

  # Add constraints to prevent overfitting
  decisionTree = tree.DecisionTreeClassifier(
      max_depth=5,              # Limit tree depth to 5 levels
      min_samples_split=20,     # Need 20+ samples to split a node
      min_samples_leaf=10       # Each leaf needs 10+ samples
  )

  decisionTree.fit(
      feature_df[[SEX_ENCODED, AGE_HEADER, PCLASS_HEADER]],
      feature_df[SURVIVED_HEADER]
  )

  Key Parameters to Try

  | Parameter         | What it does              | Good starting values |
  |-------------------|---------------------------|----------------------|
  | max_depth         | Maximum tree depth        | Try 3, 5, 7, 10      |
  | min_samples_split | Min samples to split node | Try 10, 20, 50       |
  | min_samples_leaf  | Min samples per leaf      | Try 5, 10, 20        |

  Or Use Random Forest (Better!)

  Random Forest is much more resistant to overfitting:

  from sklearn.ensemble import RandomForestClassifier

  # Random Forest averages many trees, reducing overfitting
  rf = RandomForestClassifier(
      n_estimators=100,    # Use 100 trees
      max_depth=5,         # Still constrain depth
      random_state=42      # For reproducibility
  )

  rf.fit(
      feature_df[[SEX_ENCODED, AGE_HEADER, PCLASS_HEADER]],
      feature_df[SURVIVED_HEADER]
  )

  Random Forest typically performs much better because it:
  - Trains multiple trees on random subsets of data
  - Averages their predictions
  - Reduces variance and overfitting

  Quick Test: Check Training Accuracy

  You can verify overfitting by checking training accuracy:

  from sklearn.metrics import accuracy_score

  # Predict on training data
  train_predictions = decisionTree.predict(feature_df[[SEX_ENCODED, AGE_HEADER, PCLASS_HEADER]])
  train_accuracy = accuracy_score(feature_df[SURVIVED_HEADER], train_predictions)

  print(f"Training accuracy: {train_accuracy}")

  I bet you'll see something like 0.95-1.0 training accuracy but only 0.725 test accuracy - that's the hallmark of overfitting!

  Try adding max_depth=5 to your DecisionTreeClassifier and I bet your score will improve beyond 0.765!
