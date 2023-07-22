module LA

using LinearAlgebra
using RowEchelon
using InvertedIndices
using Plots
plotly();



function cofactor(A::AbstractMatrix, T = Float64)
    ax = axes(A)
    out = similar(A, T, ax)
    for col in ax[1]
        for row in ax[2]
            out[col, row] = (-1)^(col + row) * det(A[Not(col), Not(row)])
        end
    end
    return out
end


struct DiagonalRowIndexAndConstantValue
    row::Integer
    value::Float64
end

function aug_solve(coeff_matrix::AbstractMatrix, constants_matrix::AbstractMatrix)
    return aug_solve([coeff_matrix constants_matrix])
end

function aug_solve(aug_matrix::AbstractMatrix, tol=10E-6)
    Q, R = qr(aug_matrix)
    lin_dep = minimum(abs.(diag(R)))
    rows = Integer(size(aug_matrix)[1])
    # Test this out:
    if length(diag(R)) != rows
        throw(error("Too many rows for variables!"))
    end

    function r_diagonal_value(R::Matrix, index::Integer)::Float64
        return abs(R[index, index])
    end

    diagonal = [DiagonalRowIndexAndConstantValue(row, r_diagonal_value(R, row)) for row in 1:rows]
    sort!(diagonal, by=x->x.value)
    minimums_row = first(diagonal).row
    lin_dep = first(diagonal).value
    r_minimums_constant = abs(R[minimums_row, end])


    if lin_dep < tol
        if r_minimums_constant > tol
            throw(error("Inconsistent system! No solutions! $([0 for i in rows]) | (!= 0)"))
        else
            throw(error("Dependent system! Infinite solutions! $([[0 for i in rows] 0])"))
        end
    end
    return aug_matrix[:, 1:end-1] \ aug_matrix[:,end]

end

function cramers(aug_m::AbstractMatrix)
    M = copy(aug_m[:, 1:end-1])
    xyz = copy(aug_m[:, end])
    D = det(M)
    println("D: $(D)")
    i = 1
    cols = size(M)[2]
    results = Array{Real}(undef, cols)
    while i <= cols
        D_ = copy(M)
        D_[:, i] .= xyz
        # println("D_$(i): $(D_)")
        _det = det(D_)
        println("D_$(i): det($(D_)): $(_det)")
        results[i] = _det
        i += 1
    end
    answer = Array{Real}(undef, cols)
    for j=1:cols
        answer[j] = results[j]/D
    end
    _xyz = [x for x in answer]
    return hcat(_xyz)
end

function graph_vectors(arr::AbstractVector)
    u = [[Tuple(x)] for x in arr]
    plot = quiver(Tuple([0 for x in first(u[1])]), gradient=u[1], arrowscale=0.3, headsize=1)
    for vector in u[2:end]
        
        origin = Tuple([0 for x in first(vector)])
        quiver!(origin, gradient=(vector), arrowscale=0.3, headsize=1)
    end
    return plot
end

function graph_vectors(A::AbstractMatrix)
    return graph_vectors([A[:,i] for i in 1:Integer(size(A,2))])
end


function is_LI(matrix::AbstractMatrix) 
    rows = Integer(size(matrix, 1))
    try
        aug_solve([matrix [0 for x=1:rows]])
    catch err
        if isa(err, ErrorException)
            return false
        elseif isa(err, ErrorException)
            return false
        end
    else
        return true
    end
end

function magnitude(matrix::AbstractMatrix)
    return sqrt(dot(matrix, matrix))
end

function magnitude(vector::AbstractVector)
    return magnitude(hcat(vector))
end

function inverse(matrix::AbstractMatrix)
    return (1/det(matrix))*adjugate(matrix)
end

function adjugate(M::AbstractMatrix)
    return transpose(cofactor(M))
end

function inv_ident(M::AbstractMatrix)
    rows = Integer(size(M)[1])
    return aug_solve([M identity(rows)])
end

function identity(n::Integer)
    return Matrix(I, n, n)
end

function inv_solve(aug_M::AbstractMatrix)
	coeffs_m = aug_M[:, 1:end-1]
	consts_m = aug_M[:, end]
    _inv = LinearAlgebra.inv(coeffs_m)
    return _inv * consts_m
end

function bluebrown_eigvals(M::AbstractMatrix)
	# Must be 2x2 matrix
	if Integer(size(M, 1)) != 2 && Integer(size(M, 2)) != 2
		throw("Not 2x2 Matrix")
	end
	m = tr(M)/2
	p = det(M)
	square = sqrt(m^2 - p)
	return [m + square, m - square]	
end


export cofactor, aug_solve, cramers, graph_vectors, is_LA, magnitude, inverse, adjugate, identity, bluebrown_eigvals

end