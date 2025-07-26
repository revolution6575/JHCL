function [anchor] = gen_anchor_score(X, anchor_num, views_num)

anchor = cell(views_num,1);

for v = 1:views_num
    [n,~] = size(X{v});

    X_min = min(X{v},[],1);
    x_MIN = ones(n,1)*X_min;
    X1{v} = X{v}-x_MIN;
    score = sum(X1{v}, 2);
    score(:,1) = score/max(score);
    [~,ind(1)] = max(score);

    for i=2:anchor_num
        score(:,i) = score(:,i-1).*(ones(n,1)-score(:,i-1));
        score(:,i) = score(:,i)/max(score(:,i));
        [~,ind(i)] = max(score(:,i));
    end

    ind2 = sort(ind,'ascend');
    anchor{v} = X{v}(ind2,:);
end

end