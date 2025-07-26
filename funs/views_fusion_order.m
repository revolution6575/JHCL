function [B_fusion] = views_fusion_order(B_high, v, order, alpha)

[n,m] = size(B_high{1,1});

B_fusion = zeros(n,m);

for j = 1:order
    B_fusion = B_fusion+alpha(order*(v-1)+j,1).*B_high{v,j};
end

% Normalization
B_fusion = max(B_fusion,0);
a = sum(B_fusion,2);
B_fusion = B_fusion./a;