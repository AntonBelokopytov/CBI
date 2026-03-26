function d = proc_dist(A,B)

C = A' * B;
[U,~,V] = svd(C);
W = U*V';

d = norm(A*W - B,'fro');

end
