% Evaluate prediction performance.
%
% [Input]
% class_score: (# test case) x (# class) matrix of predicted class scores.
%              Higher number represents higher confidence.
% label: (# test case) x (# class) matrix of ground truth annotations.
%        Note each case can have multiple labels
%
% [Output]
% acc: accuracy measure. Pick top prediction for each test case and 
%      see how often it matches with one of the true labels.
% f1: micro-averaged F1-score. Pick top alpha predictions for each test case,
%     calculate the contigency table for each class, sum up the table across
%     all classes then calculate the F1-score.
%
function [acc, f1] = evaluate_performace(class_score, label)
  alpha = 3;

  label = label > 0;

  [ncase nclass] = size(class_score);

  [~,o] = sort(class_score, 2, 'descend');
  p = sub2ind(size(label), (1:ncase)', o(:,1));
  acc = mean(label(p));

  a = repmat((1:ncase)', 1, alpha);
  pred = sparse(a, o(:,1:alpha), 1, size(label, 1), size(label, 2));

  tab = crosstab([0; 1; pred(:)], [0; 1; label(:)]) - eye(2);
  f1 = 2 * tab(2,2) / (2 * tab(2,2) + tab(1,2) + tab(2,1));
end
