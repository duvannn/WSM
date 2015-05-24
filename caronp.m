function matrix = caronp(mat,p)
    % implementation of caron p transform.
    % Caron, J Experiments with LSA scoring: Optimal rank and basis. 
    % In: Berry, MW eds. (2001) Computational information retrieval. SIAM, Philadelphia, PA, pp. 157-169
    dim = min(size(mat));
    res = zeros(1,dim);
    for i=1:dim
        res(i) = norm(mat(:,i),2);
    end
    m = sum(res)/dim;
    for i=1:dim
        mat(:,i) = mat(:,i)*(m/res(i))^(p);
        resu(i) = norm(mat(:,i),2);
    end
    matrix = mat;
    end
