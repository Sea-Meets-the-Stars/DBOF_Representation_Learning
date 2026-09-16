"""Which fields carry latitude in their units, and how to take it out.

Vorticity, strain and divergence all scale with the Coriolis parameter, so an
unscaled value means different dynamics at 20 degrees than at 60 and latitude
leaks into any embedding built on them.  Dividing by f turns them into
Rossby-style ratios that mean the same thing everywhere.

Kept out of both loaders so the cutout and front paths scale the same fields;
importing it pulls in nothing.  It is a package rather than a bare module
under src/ because setuptools' package discovery only installs directories --
a loose .py is importable from the working tree and missing once installed.
"""

#: Earth's rotation rate, rad/s.
OMEGA = 7.2921e-5

#: Divided by SIGNED f, so cyclonic stays positive in both hemispheres.
DIV_SIGNED = ("relative_vorticity",)

#: Divided by |f|: strain stays positive, convergence stays convergence.
DIV_ABS = ("strain_n", "strain_s", "strain_mag", "divergence")

#: Fields the front store already carries in normalised form, computed per
#: pixel before any front statistic was taken.  Prefer these to dividing a
#: stored statistic afterwards -- mean(x)/mean(f) is not mean(x/f), and the
#: two part company wherever f varies across a front's band.
NORMALISED_EQUIVALENT = {"relative_vorticity": "rossby_number"}
