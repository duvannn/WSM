function matrix = caronp(mat,p)
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
