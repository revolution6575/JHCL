%% Data disruption
function [X_disrupt, y_disrupt] = disrupt_data(X, y, views_num, best_view, disrupt_index_all)

X_disrupt = cell(1,views_num);
disrupt_num = size(disrupt_index_all,2);

for i = 1:views_num
    disrupt_index = disrupt_index_all(1:views_num,:);
    disrupt_index_in = disrupt_index_all(views_num+1:2*views_num,:);

%         firstRow = disrupt_index(1, :);
%         disrupt_index = repmat(firstRow, views_num, 1);

    X_disrupt{i} = X{i};

    for j = 1:disrupt_num
        X_disrupt{i}(disrupt_index(i,j),:) = X{i}(disrupt_index(i,disrupt_index_in(i,j)),:);
    end
end

y_disrupt = y;

for i = 1:disrupt_num
    y_disrupt(disrupt_index(best_view,i),:)  = y(disrupt_index(best_view,disrupt_index_in(best_view,i)),:);
end
