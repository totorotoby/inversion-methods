using Printf
using ForwardDiff
using Plots
using SparseArrays
using LinearAlgebra
using DataStructures

### gaussian integration of func1 x func2 (interpolating basis function i and j on nodes) x param
# weights and nodes pulled from: https://pomax.github.io/bezierinfo/legendre-gauss.html
function gauss_integrate(func1, func2, i, param, j, nodes)

    weights = [0.6521451548625461
               0.6521451548625461
               0.3478548451374538
               0.3478548451374538]

    abscissa = [-0.3399810435848563
    	        0.3399810435848563
    	        -0.8611363115940526
    	        0.8611363115940526]

    val = 0.0
    scale = (nodes[end] - nodes[1]) * .5
    c = (nodes[end] + nodes[1]) * .5

    for l in 1:length(weights)
        val += weights[l] * func1(scale * abscissa[l] + c, i, nodes) *
            param(scale * abscissa[l] + c) * 
            func2(scale * abscissa[l] + c, j, nodes)
    end

    return scale *  val
    
end


# if computing adjoint adj = 1 else -1
function assemble_stiffness!(adj, Ne, Nbasis, p, x, k, a, I, J, V; mod_sparsity=true)

    nstart = 1
    for e in 1:Ne
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            for j in 1:Nbasis
                col = (p*e) + (j-p)

                # in the forward problem if we know k (or a??) we can just plug it in here
                # in the inverse problem we don't know it so we start with a guess here
                # guessing a one function for now
                v_diff = gauss_integrate(dlb, dlb, i, k, j, x[nstart : nstart + p])
                v_adv = gauss_integrate(lb, dlb, i, a, j, x[nstart : nstart + p])

                idx = in_COO(I, J, row, col)
                
                if mod_sparsity
                    if idx > 0 
                        V[idx] += (v_diff + adj * v_adv)
                    else
                        push!(I, row)
                        push!(J, col)
                        push!(V, v_diff + adj * v_adv)
                    end
                else
                    V[idx] += (v_diff + adj * v_adv)
                end
            end
        end
        nstart += p
    end
end

function assemble_forcing!(Ne, Nbasis, p, x, forcing, F)
    nstart = 1
    # global stiffness matrix assembly
    for e in 1:Ne
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            F[row] += gauss_integrate(lb, one, i, forcing, i, x[nstart : nstart + p])
        end
        nstart += p
    end
end


# need to generalize to neumann, etc...
function enforce_boundary!(A, F)
    A[1, 1] = 1.0
    A[1, 2:end] .= 0.0
    A[end, end] = 1.0
    A[end, 1:(end-1)] .= 0.0
    F[1] = 0.0
    F[end] = 0.0
end

function in_COO(I, J, i, j)
    for idx in 1:length(I)
        if I[idx] == i && J[idx] == j
            return idx
        end
    end
    return -1
end

#---- Barycentric lagragian interpolation ----#

# computes numerator
function lag(x, nodes)
    l = 1
    for i in 1:length(nodes)
        l *= (x - nodes[i])
    end
    return l
end

# derivative of numerator for weights
dlag(x, nodes) = ForwardDiff.derivative(x -> lag(x, nodes), x)

# evaluate basis function j at x with nodes
function lb(x, j, nodes)

    l = lag(x, nodes)
    w = 1/dlag(nodes[j], nodes)
    
    if x != nodes[j]
        return (l * w)/(x - nodes[j])
    else
        return 1.0
    end
end

# basis function derivative
dlb(x, j, nodes) = ForwardDiff.derivative(x -> lb(x, j, nodes), x)

#---- model parameters and test parameters ----#

one(x, j, n) = 1.0
one(x) = 1.0

# parameters for testing
# second set is without any advection, or variable coefficents
forcing_exact(x) = x - mms(x) #1.0 
forcing(x) = x
mms(x) = 1/4*(x^2 - 2x^4) #0.0
k_exact(x) = 1/x #1.0
a_exact(x) = 0  # 0.0
u_exact(x) = -1/8 * x^4 + 1/8 * x^2 # #-1/2 * x^2 + 5*x

# p order lagrangian basis expansion with current coords at x
function expansion(x, p, coords, n_global)

    # get local nodes, and local coordinates
    e, n_local = XToN(x, p, n_global)
    coords_local = coords[1 + (e-1) * p : 1 + e*p]
    eval = 0
    for i in 1:p+1
        eval += coords_local[i] * lb(x, i, n_local)
    end
    
    return eval
end

# given point in domain, which element (nodes in element) is it in
function XToN(x, p, nodes)
    
    elements = nodes[1:p:end]
    e = searchsortedfirst(elements, x)
    e = e == 1 ? 1 : e - 1
    
    return e, nodes[1 + (e-1) * p : 1 + e*p]
end

let

    # number of elements
    Ne = 100
    # basis order
    p = 1
    # number of nodes
    N = p*Ne + 1
    # domain boudarys [L, R]
    L = 0
    R = 1
    # length of element
    h = (R-L)/(N-1)
    # nodes
    x = collect(L:h:R)
    xfine = collect(L:.01:R)
    # number basis functions
    Nbasis = p + 1

    # basis functions and integrator sanity check
    # @assert isapprox(gauss_integrate(lb, lb, 1, integration_test, 2, [-1.0, 0.0 , 1.0]), -2/15, atol=1e-16)
    # @assert isapprox(gauss_integrate(dlb, dlb, 1, integration_test, 3, [-1.0, 0.0 , 1.0]), 0.0, atol=1e-16)

    #---- testing forward model ----#

    #=
    # COO for global matrix
    I = Int64[]
    J = Int64[]
    V_test = Float64[]

    #stiffness matrix
    assemble_stiffness!(-1, Ne, Nbasis, p, x, k_exact, a_exact, I, J, V_test)
    A_test = sparse(I, J, V_test, N, N)

    # forcing vector
    F = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, x, forcing_exact, F)
    enforce_boundary!(A_test, F)
    
    # sparsity pattern sanity check
    # display(spy(A_test))
    
    # convergence sanity check
    u = (A_test\F)
    ue = u_exact.(x)
    error = abs.(ue - u)
    plot(x, u, label="numerical")
    display(plot!(x, ue, label="exact"))
    display(plot!(x, error, label="error"))
    =#
    
    #---- Solving inverse problem ----#

    I = Int64[]
    J = Int64[]
    V_forward = Float64[]

    # assemble forward and adjoint stiffness
    assemble_stiffness!(-1, Ne, Nbasis, p, x, one, a_exact, I, J, V_forward)
    V_adjoint = zeros(length(I))
    assemble_stiffness!(1, Ne, Nbasis, p, x, one, a_exact, I, J, V_adjoint, mod_sparsity=false)
    A_forward = sparse(I, J, V_forward, N, N)
    A_adjoint = sparse(I, J, V_adjoint, N, N)

    # forcing vector
    F_forward = zeros(N)
    F_adjoint = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, x, forcing, F_forward)
    enforce_boundary!(A_forward, F_forward)


    # initial guess 
    k_iter = ones(N)
    u_iter = zeros(N)
    u_adjoint = zeros(N)
    u_data = u_exact.(x)
    u_error = zeros(N)
    ue =  u_exact.(x)
    
    # what is a good stopping criteria here?
    descent_iter = 1
    for i in 1:descent_iter

        # forward solve
        u_iter .= A_forward\F_forward
        u_error .= u_iter - u_data
        
        assemble_forcing!(Ne,
                          Nbasis,
                          p,
                          x,
                          val -> expansion(val, p, u_error, x),
                          F_adjoint)
        
        enforce_boundary!(A_adjoint, F_adjoint)
        u_adjoint .= A_adjoint\F_adjoint

        
        # alternatively (and would save a lot of memory) the adjoint operator is just A_forward transpose,
        # and can instead  be used.
        # enforce_boundary!(A_forward', F_adjoint)
        # u_adjoint2 = (A_forward')\F_adjoint
        # display(plot(x, u_adjoint2, label="transpose adjoint"))

        
        plot(x, u_error, label="error")
        plot!(x, u_iter, label="estimate")
        plot!(x, ue, label="exact")
        display(plot!(x,u_adjoint, label="adjoint"))
        
    end
    nothing
    
end
