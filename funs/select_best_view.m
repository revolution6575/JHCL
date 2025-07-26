function[best_view]=select_best_view(X, y, views_num, cluster_num, select_best_view_style)

sill = [];

for v = 1: views_num

     D{v} = L2_distance_1(X{v}',X{v}');

    if select_best_view_style == 1

        [index{v},~,~]=kmeans(D{v},cluster_num,'maxiter',1000,'replicates',20,'EmptyAction','singleton');

    elseif select_best_view_style ==2
        X_CLR{1} = X{v};

        a = max(X_CLR{1}(:));
        X_CLR{1} = double(X_CLR{1}./a);

        anchor_rate=0.1:0.1:1;

        c = length(unique(y));
        opt1. style = 1;
        opt1. IterMax =50;
        opt1. toy = 0;

        [~, ~, index{v}] = FastmultiCLR(X_CLR, c, anchor_rate(2), opt1,1);

    end
        sill = [sill,mean(silhouette(D{v},index{v},'cityblock'))];

end

[~,best_view] = max(sill);

end