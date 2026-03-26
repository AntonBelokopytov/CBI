function C = upper2cov(v)

n = (-1 + sqrt(1 + 8 * numel(v))) / 2;

C = zeros(n);
upper_mask = triu(true(n));
C(upper_mask) = v;

upper_triu_mask = triu(true(n), 1);
C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);

C = C + triu(C, 1)';

end
