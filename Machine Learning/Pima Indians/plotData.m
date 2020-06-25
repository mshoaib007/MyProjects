function plotData(X,y)
figure; hold on;
positive=find(y==1);
negative=find(y==0);
% bar(X(positive,1),X(positive,2),X(positive,3),X(positive,4),X(positive,5),X(positive,6),X(positive,7),X(positive,8),...
%      'k+','LineWidth',8,'MarkerSize',7);
% bar(X(negative,1),X(negative,2),X(negative,3),X(negative,4),X(negative,5),X(negative,6),X(negative,7),X(negative,8),...
%      'ko','MarkerFaceColor','y','MarkerSize',7);

plot(X(positive,1),X(positive,2),X(positive,3),X(positive,4),X(positive,5),X(positive,6),X(positive,7),X(positive,8),...
    'k+','LineWidth', 2,'MarkerSize',7);

plot(X(negative,1),X(negative,2),X(negative,3),X(negative,4),X(negative,5),X(negative,6),X(negative,7),X(negative,8),...
   'ko', 'MarkerFaceColor', 'y', ... 
'MarkerSize', 7);

hold off;
end
