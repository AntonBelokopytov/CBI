function W = procrustes_rotation(X,Y)

C = X' * Y;
[U,~,V] = svd(C);
W = U*V';

end
