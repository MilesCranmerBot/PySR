using DocumenterVitepress

# Whether this tag is the newest release, and so owns the `stable` alias.
#
# Julia parses Python-style prerelease tags such as v2.0.0a2 as v2.0.0-a2, and
# determine_bases then reports the alpha as owning the v2.0.0 release folder, so only
# plain release tags reach it.
function claims_stable(subfolder; kwargs...)
    occursin(r"^v\d+(\.\d+){0,2}$", subfolder) || return false
    return "stable" in DocumenterVitepress.determine_bases(subfolder; keep = :patch, kwargs...)
end
