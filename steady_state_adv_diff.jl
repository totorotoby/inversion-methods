using Printf
using ForwardDiff
using Plots
using SparseArrays
using LinearAlgebra
using Statistics
using DataStructures

### gaussian integration of funcs multiplied together with args for each function
# weights and abscissa pulled from: https://pomax.github.io/bezierinfo/legendre-gauss.html
function gauss_integrate(element, funcs...)

    weights = [0.6521451548625461
               0.6521451548625461
               0.3478548451374538
               0.3478548451374538]

    abscissa = [-0.3399810435848563
    	        0.3399810435848563
    	        -0.8611363115940526
    	        0.8611363115940526]

    val = 0.0
    scale = (element[end] - element[1]) * .5
    c = (element[end] + element[1]) * .5

    for l in 1:length(weights)
        val += weights[l] * 
            reduce(*, [f(scale * abscissa[l] + c) for f in funcs])
        #@show reduce(*, [f(scale * abscissa[l] + c) for f in funcs])
    end
    return scale *  val
end

# Wrapper for gauss_integrate that integrates the whole domain x
function domain_integrate!(Ne, Nbasis, p, x, u, funcs...)
    
    for e in 1:Ne
        nodes = EToN(e, p, x)
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            u[row] += gauss_integrate(nodes, funcs...)
        end
    end
end

#=
This function assembles a discrete diffusion and advection operator from the basis functions:
    Ne: number of elements
    Nbasis: number of basis functions per element = p + 1 (might not need to be carrying this around
    p: order of basis
    func1: function to integrate in element (a set of basis functions or its derivative)
    func2: same as func1
    k: parameter, can be known or
       a guess if doing the inverse problem
    I: non zero row indices
    J: non zero column indices
    V: non zero values
=#
function assemble_matrix!(Ne, Nbasis, p,
                          x, func1, func2, k,
                          I, J, V)
    for e in 1:Ne
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            for j in 1:Nbasis
                col = (p*e) + (j-p)

                # in the forward problem if we know k (or a??) we can just plug it in here
                # in the inverse problem we don't know it so we start with a guess here
                # guessing a one function for now
                nodes = EToN(e, p, x)
                v = gauss_integrate(nodes, x -> func1(x, i, nodes) , x ->  func2(x, j, nodes), k)
                idx = in_COO(I, J, row, col)
                
                if idx > 0 
                    V[idx] += v
                else
                    push!(I, row)
                    push!(J, col)
                    push!(V, v)
                end
            end
        end
    end
end

function assemble_forcing!(Ne, Nbasis, func, p, x, forcing, F)
    nstart = 1
    # global stiffness matrix assembly
    for e in 1:Ne
        nodes = EToN(e, p, x)
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            F[row] += gauss_integrate(nodes, x -> func(x, i, nodes), forcing, one)
        end
        nstart += p
    end
end

function assemble_forcing!(Ne, Nbasis, p, x, forcing, F)
    nstart = 1
    # global stiffness matrix assembly
    for e in 1:Ne
        nodes = EToN(e, p, x)
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            F[row] += gauss_integrate(nodes, forcing, one)
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

# evaluate basis function local index j at x with element nodes "nodes"
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
forcing_exact(x) = 1.0 
forcing(x) = 1.0
# mms(x) = 1/4*(x^2 - 2x^4)
k_exact(x) = 1.0
a_exact(x) = 0.0
u_exact(x) = (1/(2 .* k_exact.(x))) * (x - x^2)

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

#=
 Element to Nodes:
 e - element number
 p - order of basis
 nodes - list of global nodes
 returns nodes in element =#
EToN(e, p, nodes) = nodes[(e-1)*p + 1 : (e-1)*p + p + 1]

let
    
    # number of elements
    Ne = 255
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

    
    # COO for global matrix
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    #stiffness matrix
    assemble_matrix!(Ne, Nbasis, p, x, dlb, dlb, k_exact, I, J, Vdiff)
    #Vadv = zeros(length(Vdiff))
    #assemble_matrix!(Ne, Nbasis, p, x, lb, dlb, a_exact, I, J, Vadv)
    A_test = sparse(I, J, Vdiff, N, N)

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
    plot!(x, ue, label="exact")
    display(plot!(x, error, label="error"))

    #---- Plot basis ----#
    #=
    plt = plot()
    for e in 1:Ne
        for i in 1:p+1
            e_nodes = EToN(e, p, x)
            llb = (x -> lb(x, i, e_nodes))
            xfine = e_nodes[1]:(e_nodes[end] - e_nodes[1])/100:e_nodes[end]
            plt = plot!(xfine, llb.(xfine), legend=false)
        end
    end
    display(plt)
    =#
    
    #---- Sensitivity equations ----#
    #=
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    assemble_matrix!(Ne, Nbasis, p, x, dlb, dlb, k_exact, I, J, Vdiff)
    Vadv = zeros(length(Vdiff))
    assemble_matrix!(Ne, Nbasis, p, x, lb, dlb, a_exact, I, J, Vadv)
    A_forward = sparse(I, J, Vdiff - Vadv, N, N)
    F_sense = zeros(N)

    plt = plot()
    for e in 1:1
        for i in 1:1
            for j in 1:1
                e_nodes = EToN(e,p, x)
                integral = gauss_integrate(e, (x -> lb(x, i, e_nodes)), (x -> lb(x, j, e_nodes)))
                llb1 = (x -> lb(x, i, e_nodes))
                llb2 = (x -> lb(x, j, e_nodes))
                xfine = e_nodes[1]:(e_nodes[end] - e_nodes[1])/100:e_nodes[end]
                plt = plot!(xfine, llb1.(xfine), legend=false)
                plt = plot!(xfine, llb2.(xfine), legend=false)
                @show e, i, j, integral
            end
        end
    end
    display(plt)
    =#
    #assemble_forcing!(Ne, Nbasis, p, x, forcing, F_forward)
#end

    #=
    #---- Solving inverse problem ----#

    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]

    # assemble forward and adjoint stiffness
    assemble_matrix!(Ne, Nbasis, p,
                     x, dlb, dlb, one,
                     I, J, Vdiff)

    Vadv = zeros(length(Vdiff))
    Vgrad = zeros(length(Vdiff))
    Vmass = zeros(length(Vdiff))
    
    assemble_matrix!(Ne, Nbasis, p,
                     x, lb, dlb, a_exact,
                     I, J, Vadv)

    assemble_matrix!(Ne, Nbasis, p,
                     x, lb, dlb, one,
                     I, J, Vgrad)

    assemble_matrix!(Ne, Nbasis, p,
                     x, lb, lb, one,
                     I, J, Vmass)

    M = sparse(I, J, Vmass, N, N)
    D = sparse(I, J, Vgrad, N, N)

    
    A_forward = sparse(I, J, Vdiff - Vadv, N, N)
    A_adjoint = sparse(I, J, Vdiff + Vadv, N, N)

    # forcing vector
    F_forward = zeros(N)
    F_adjoint = zeros(N
    assemble_forcing!(Ne, Nbasis, p, x, forcing, F_forward)
    enforce_boundary!(A_forward, F_forward)

    # initial guess
    k_iter = ones(N)
    u_iter = zeros(N)
    u_iter_grad = zeros(N)
    u_adjoint_grad = zeros(N)
    u_adjoint = zeros(N)
    u_data = u_exact.(x)
    u_error = zeros(N)
    dJdk = zeros(N)
    ue =  u_exact.(x)

    step_size = .001
    # what is a good stopping criteria here?
    descent_iter = 10
    for i in 1:descent_iter

        # forward solve
        u_iter .= (A_forward)\F_forward
        u_error .= u_iter - u_data

        #plot!(x, u_error, label="error")
        #plot!(x, u_iter, label="estimate")
        #display(plot!(x, u_data, label="exact"))
        
        assemble_forcing!(Ne,
                          Nbasis,
                          p,
                          x,
                          val -> expansion(val, p, u_error, x),
                          F_adjoint)
        
        enforce_boundary!(A_adjoint, F_adjoint)
        u_adjoint .= (A_adjoint)\F_adjoint
        u_iter_grad .= M\(D * u_iter)
        u_adjoint_grad .= M\(D * u_adjoint)

        dJdk .= u_adjoint_grad .* u_iter_grad
        #=
        domain_integrate!(Ne, Nbasis, p, x, dJdk,
                          val -> expansion(val, p, u_iter_grad, x),
                          val -> expansion(val, p, u_adjoint_grad, x))

        =#

        k_iter .= k_iter .- step_size * dJdk

        # reassemble forward and adjoint operators (this is probably a horrible way of doing this...)
        Vdiff .= 0
        assemble_matrix!(Ne, Nbasis, p,
                         x, dlb, dlb, val -> expansion(val, p, k_iter, x),
                         I, J, Vdiff)

        A_forward.nzval .= Vdiff - Vadv
        A_adjoint.nzval .= Vdiff + Vadv
        enforce_boundary!(A_forward, F_forward)
        enforce_boundary!(A_adjoint, F_adjoint)
        #display(A_forward)
        
        
        # domain_integrate!(Ne, Nbasis, p, x, dJdk,
        #                   val -> expansion(val, p, u_iter_grad, x),
        #                   val -> expansion(val, p, u_adjoint_grad, x))
        
        # domain_integrate!(Ne, Nbasis, p, x, dJdk, val -> expansion(val, p, u_iter_grad, x),
        #                 val -> expansion(val, p, u_adjoint_grad, x),
        #                 (val, node)-> lb(val, node, x))
            
        # alternatively (and would save a lot of memory) the adjoint operator is just A_forward transpose,
        # and can instead  be used.
        # enforce_boundary!(A_forward', F_adjoint)
        # u_adjoint2 = (A_forward')\F_adjoint
        # display(plot(x, u_adjoint2, label="transpose adjoint"))


        p1 = plot(x, u_error, label="error")
        plot!(p1, x, u_iter, label="estimate")
        plot!(p1, x, ue, label="exact")
        p4 = plot(x, u_adjoint, label="adjoint")
        
        p2 = plot(x, dJdk, label="dJdk")
        plot!(p2, x, u_iter_grad, label="u_iter_grad")
        plot!(p2, x, u_adjoint_grad, label="u_adjoint_grad")
        p3 = plot(x, k_iter, label="k")
        display(plot(p1, p3, p2, p4))
        #sleep(.01)
    end
    =#
    
    nothing

end
