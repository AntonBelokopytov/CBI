function [W, A, corrs] = espoc(X_epochs, Z, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'X_min_var_explained', 1, ...
                  'whitening_reg', 10e-5, ...
                  'cca_mode', 'regularized', ...
                  'cca_reg', 10e-5);

Z = (Z - mean(Z,2)) ./ std(Z,[],2);

% Xraw = [];
% for i=1:size(X_epochs,3)
%     Xraw = [Xraw,X_epochs(:,:,i)'];
% end
% Xraw = Xraw - mean(Xraw,2);

% ---
[Feat, Cxx, Epochs_cov] = get_covariance_series(X_epochs);
[Featdr, Uf] = project_to_pc(Feat, opt.X_min_var_explained);
Cff = cov(Feat');

if strcmp(opt.cca_mode, 'regularized')
    [Vfdr, Vz] = cca(Featdr', Z', opt);
elseif strcmp(opt.cca_mode, 'standard') 
    [Vfdr, Vz] = canoncorr(Featdr', Z');
end
% Return found filters from dimension reduced space 
Vfw = Uf * Vfdr;
Afw = Cff * Vfw;

% en = Vfw' * Feat;
% en = (en - mean(en)) / std(en);
% plot(en)
% hold on
% plot(Z)

% Project and normalize EEG/MEG filters
for global_src_idx=1:size(Vfw,2)
    [w, a, s] = project_to_manifold(Vfw(:,global_src_idx), Afw(:,global_src_idx), Cxx);
    
    % Project target variable to its CCA component 
    Zpr = Vz(:,global_src_idx)'*Z;
    
    % Find correlation of the filters
    Env = [];
    for local_src_idx=1:size(w,2)
        for ep_idx=1:size(Epochs_cov,3)
            Env(ep_idx) = w(:,local_src_idx)' * Epochs_cov(:,:,ep_idx) * w(:,local_src_idx);
        end
        cr(local_src_idx)=corr(Env',Zpr');
    end
    % [cr,idx] = sort(cr,'descend');
    % w = w(:,idx);
    % a = a(:,idx);
    
    eigenvalues(global_src_idx,:) = s;
    corrs(global_src_idx,:) = cr;
    W(global_src_idx,:,:) = w;
    A(global_src_idx,:,:) = a;
end

if size(W,1)==1
    corrs = squeeze(corrs);
    W = squeeze(W);
    A = squeeze(A);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [v] = cov2upper(C)

upper_triu_mask = triu(true(size(C)),1);
upper_mask = triu(true(size(C)));
C(upper_triu_mask) = C(upper_triu_mask)*sqrt(2);
upper_triangle = C(upper_mask);
v = upper_triangle(:);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C = upper2cov(v)

n = (-1 + sqrt(1 + 8 * numel(v))) / 2;

C = zeros(n);
upper_mask = triu(true(n));
C(upper_mask) = v;

upper_triu_mask = triu(true(n), 1);
C(upper_triu_mask) = C(upper_triu_mask) / sqrt(2);

C = C + triu(C, 1)';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [F, Cxx, Epochs_cov] = get_covariance_series(X_epochs)

% Function to get upper triangular covarience time series in dimension
% reduced space

[~,n_channels,n_epochs] = size(X_epochs);
n_features = (n_channels^2-n_channels)/2+n_channels;

Epochs_cov = zeros(n_channels,n_channels,n_epochs);
for ep_idx = 1:n_epochs
    Xcov = cov(X_epochs(:,:,ep_idx));
    Epochs_cov(:,:,ep_idx) = Xcov;
end
% Mean covariance matrix
Cxx = mean(Epochs_cov,3);
Wm = eye(size(Cxx,1)) / sqrtm(regularize(Cxx));

% Whightened covariance series (upper triangular parts)
F = zeros(n_features,n_epochs);
for ep_idx = 1:n_epochs
    Xcov = Epochs_cov(:,:,ep_idx);
    F(:, ep_idx) = cov2upper(Xcov);
end

F = F - mean(F,2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X_proj, U] = project_to_pc(X, min_var_explained)

X = X - mean(X,2);

[U,S,~] = svd(X,"econ");

S = diag(S);

tol_rank = max(size(X)) * eps(S(1));
r = sum(S > tol_rank);

ve = S.^2;
var_explained = cumsum(ve) / sum(ve);
var_explained(end) = 1;

tol = 1e-12;
n_components = find(var_explained >= min_var_explained - tol, 1);
if isempty(n_components)
    n_components = r;
end

U = U(:,1:n_components);

% Project
X_proj = U' * X;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Vx, Vy, Cxx, Cyy] = cca(X, Y, opt)

% Regularized CCA

gamma = opt.cca_reg;

X = X - mean(X,1);  
Y = Y - mean(Y,1);

[n,~] = size(X);

Cxx = (X' * X) / (n-1);
Cyy = (Y' * Y) / (n-1);
Cxy = (X' * Y) / (n-1);

scale_x = trace(Cxx) / size(Cxx,1);
scale_y = trace(Cyy) / size(Cyy,1);
Sxx_r = (1-gamma)*Cxx + gamma*scale_x*eye(size(Cxx));
Syy_r = (1-gamma)*Cyy + gamma*scale_y*eye(size(Cyy));
Sxx_r = (Sxx_r + Sxx_r') / 2; 
Syy_r = (Syy_r + Syy_r') / 2; 

Rx = chol(Sxx_r,'upper');
Ry = chol(Syy_r,'upper');

K = Rx' \ (Cxy / Ry);            
[Ux,~,Uy] = svd(K,'econ');

Vx = Rx \ Ux; 
Vy = Ry \ Uy; 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W, A, s] = project_to_manifold(Vf,Af,Cxx)    
    Cxx_r = regularize(Cxx);
    
    WW = upper2cov(Vf);
    WW = (WW + WW') / 2;

    AA = upper2cov(Af);
    AA = (AA + AA') / 2;
    
    [Uw, S] = eig(AA, Cxx_r);    
    % [Uw, S] = eig(WW);    

    % Wm = eye(size(Cxx,1)) / sqrtm(regularize(Cxx));

    [s, idxs] = sort(diag(S),'descend');
    Uw = Uw(:,idxs);
    
    n_channels = size(Cxx, 1);
    n_local_src = size(Uw, 2);
    W = zeros(n_channels, n_local_src);
    A = zeros(n_channels, n_local_src);
    
    for local_src_idx = 1:n_local_src
        wi = Uw(:, local_src_idx); 
        
        Wprn = wi / sqrt(wi' * Cxx * wi);
        
        W(:, local_src_idx) = Wprn;
        A(:, local_src_idx) = Cxx_r * Wprn; 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C_r = regularize(C)    
    C_r = C + 10e-5 * eye(size(C)) * trace(C)/size(C,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER HELPERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function opt = propertylist2struct(varargin)
% PROPERTYLIST2STRUCT - Make options structure from parameter/value list
%
%   OPT= propertylist2struct('param1', VALUE1, 'param2', VALUE2, ...)
%   Generate a structure OPT with fields 'param1' set to value VALUE1, field
%   'param2' set to value VALUE2, and so forth.
%
%   See also set_defaults

opt= [];
if nargin==0,
  return;
end

if isstruct(varargin{1}) | isempty(varargin{1}),
  % First input argument is already a structure: Start with that, write
  % the additional fields
  opt= varargin{1};
  iListOffset= 1;
else
  % First argument is not a structure: Assume this is the start of the
  % parameter/value list
  iListOffset = 0;
end

nFields= (nargin-iListOffset)/2;
if nFields~=round(nFields),
  error('Invalid parameter/value list');
end

for ff= 1:nFields,
  fld = varargin{iListOffset+2*ff-1};
  if ~ischar(fld),
    error('Invalid parameter/value list');
  end
%  prp= varargin{iListOffset+2*ff};
%  opt= setfield(opt, fld, prp);
  opt.(fld)= varargin{iListOffset+2*ff};
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [opt, isdefault]= set_defaults(opt, varargin)
%[opt, isdefault]= set_defaults(opt, field/value list)
%
%Description:
% This functions fills in the given struct opt some new fields with
% default values, but only when these fields DO NOT exist before in opt.
% Existing fields are kept with their original values.
%
%Example:
%   opt= set_defaults(opt, 'color','g', 'linewidth',3);
%
% The second output argument isdefault is a struct with the same fields
% as the returned opt, where each field has a boolean value indicating
% whether or not the default value was inserted in opt for that field.

% blanker@cs.tu-berlin.de

% Set 'isdefault' to ones for the field already present in 'opt'
isdefault= [];
if ~isempty(opt),
  for Fld=fieldnames(opt)',
    isdefault.(Fld{1})= 0;
  end
end

defopt = propertylist2struct(varargin{:});
for Fld= fieldnames(defopt)',
  fld= Fld{1};
  if ~isfield(opt, fld),
    %% if opt is a struct *array*, the fields of all elements need to
    %% be set. This is done with the 'deal' function.
    [opt.(fld)]= deal(defopt.(fld));
    isdefault.(fld)= 1;
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
