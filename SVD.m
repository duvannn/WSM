function matrix  =  SVD(file)
    % the method removes the first rd dimensions 
    mat = load(strcat(file,'.mat'));
    i=double(mat.i);
    j=double(mat.j);
    s=double(mat.s);
    matrix = sparse(i,j,s);
    thetime = tic();
    [ U, S, T] = svds(matrix,d);
    telapsed = toc(thetime);
    disp('svd-done'); disp(telapsed);
    matrix = U*S;
end