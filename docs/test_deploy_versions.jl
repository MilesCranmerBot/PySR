using Test

include(joinpath(@__DIR__, "deploy_versions.jl"))

@testset "stable ownership" begin
    tags = [v"2.1.0", v"2.0.0", v"1.5.10"]

    @test claims_stable("v2.1.0"; all_tagged_versions = tags, log = false)
    @test !claims_stable("v2.0.0"; all_tagged_versions = tags, log = false)
    # determine_bases would report this alpha as owning v2.0.0
    @test !claims_stable("v2.0.0a2"; all_tagged_versions = tags, log = false)
    @test !claims_stable("dev"; all_tagged_versions = tags, log = false)
    @test !claims_stable("previews/PR123"; all_tagged_versions = tags, log = false)
end
