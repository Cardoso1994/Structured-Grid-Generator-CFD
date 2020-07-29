# Aerodynamics

## Basics
Aerodynamics is a subfield of Fluid Mechanics that particularly deals with the
interaction between bodies and airflows, and the forces that are generated
thanks to that interaction. Although now, aerodynamics is a term that is
commonly used to refer to other fields of fluid mechanics with no difference at
all.

There are two kinds of flows of interest in aerodynamics and fluid dynamics,
one is external flows, in these flows a body is inmerse inside an air flow. The
second type of flows are internal flows, in this kind of flows a fluid flow
runs inside a duct, tube, etc.

A fluid behaves a lot different than a solid body, this due to the structure
that made them at a molecular level, in a solid the particles a very close
together, thus the molecular forces are strong, the opposite occurs with fluids
(liquids and gases) where their particles are far apart each other givig as a
result that the molecular forces are weaker. The consequence of this is that
solids behave as a rigid body, meaning for example that if we say that a solid
is moving at a given velocity, all of its particles are moving at that
velocity. In a fluid that is not the case because a consequence of the weak
forces between the particles that made them is that they have a "squishy"
behaviour, so there might be particles moving at different velocoties and in
different directions across the fluid. In other words, many properties can have
different values depending on which part of the flow they are being measured.
This properties, for example, are, density, velocity, temperature. To this kind
of properties we refer to as "POINT PROPERTIES".

### Pressure and Shear stress
As it has already been said, aerodynamics deals with airflows and body
interaction, and the forces produced during that interaction. In aeronautical
engineering we are more concerned with external flows, such as an airfoil,
wing, fuselage or a complete aircraft conguration inside a freestream air flow
current. It is important to say that EVERY aerodynamic force produced in a
fluid-body interaction is product only of two distributions, namely pressure
distribution and shear stress distribution around the body's surface. It does
not matter how complex the flow might be, or how complex the geometry of the
body is, every force is a product of this two distributions. Both of this
parameters as it has already been said, act on the body surface, hence they
have units of force per area units (N / m²), pressure acts in normal direction
to the surface whereas shear stress acts in the direction tangent to the
surface.

The pressure distribution and shear stress distribution can be computed and
expressed as a unique "Resultant" Force. One can also express this force in
terms of other forces, the most common being "Lift and Drag", lift is the
resulting force in perpedicular direction to the flow stream (positive upwards)
and drag is the resulting force in the flow direction (positive in the
direction of flow). One can also decompose the force in a "Normal and Axial"
forces, these forces are perpendicular and axial, respectively, to the
aerodynamic chord of the body (the arodynamic chord is a characteristic length
of the body, in an airfoil is the imaginary line that joins the leading edge
with the trailing edge). A natural question to come at this point is, where in
the airfoil are those forces (normal and axial) should be placed?. The right
answer is that these resultant forces should be placed in a position where they
produce the exact same "moment" around the leading edge than that produced by
the distributions of pressure and shear stress. This point is known as " center
of pressure". Another characteristic of this point is that, if we compute the
moment around this point, it should be equal to zero, hence that is another
common definition of the center of pressure, 'the point in which the
aerodynamic moment around it is zero'.

### Flow similarity
Flow similarity is a concept that implies that two flows may have similar
dimensionless properties, and hence, similar behaviours. Flow similiraty
concepts are derived from the "Buckingham Pi" theorem for dimesnional analysis.
Buckingham Pi theorem must be studied apart and in detail.

As a consequence of the dimensional analysis, we know that the resultant
aerodynamic (R) force can be expressed in terms of a dimensionless parameter
C_{R}, and this parameter is only a functions of two properties of the airflow
of the freestream, the Reynolds number and the Mach number.

The Reynolds number is another dimensionless parameter that exress the relation
of inertial forces to viscous forces, the bigger Re the more effect of inertial
forces is present, in other words the flow is becomes more turbulent as the
Reynolds number increases, when the Reynolds number is low, the viscous effects
are of bigger importance that the inertial ones. On the other hand, the Mach
number is a relation between the freestream velocity and the velocity of sound
for a given set of air conditions.

As it has been stated, the Cr coefficient is a function of Re and M, and
because the resultan force R can be expressed in Lift and Drag, both L and D
can be expressed in terms of their own coefficients Cl and Cd. Furthermore, as
the force R creates a moment around a certain point, the aerodynamic moment M
can also be expressed in terms of a coefficient Cm. All of this coefficients
will be in fact, functions only of the Reynolds number Re and the Mach number
M.

All of this is very important to note, because it is the knowledge in which
wind tunnel simulation is based on. A flow is similar to another, or flow
similarity is achieved when:
- The streamlines patterns of the flow are geometrically similar
- The relations of the properties between flows (T / Tinf, P / Pinf, V /
    Vinf, etc) are the same when plotted in nondimensional coordinates
- The force coefficients are the same

In fact, the 3rd point is just a consequence of the 2nd point. Put in simpler
words, two flows are similar if
- They bodies inmerse in the fluid, as well as any other solid boundary, are
    geometrically similar
- The Mach and Reynolds numbers are both the same for both flows.

### Types of flow
#### Continuum vs Free molecule flow
This is nothing more than a distinction between the kinds of the interaction of
a particular flow with external bodies. Let's call the D to the distance
between particles of air, if the body with which these particles are
interacting is in many orders of magnitude bigger than the distance D, we can
say that the fluid in motion behaves as a continuum. In contrast, if the size
of a given body in interaction with this flow is in the same order of
magnitude, we talk about a free molecule flow, this is because any interaction
between a given particle and the body will occur every now and then, and they
will not be frequent.

#### Inviscid vs Viscous flow
The distinction between an inviscid and a viscous flow is that in an inviscid
flow, we can neglect the effects of the viscosity of the fluid, this is mainly
because the contribution of the viscous effects to the resultant force is very
small to become important, nevertheless, there is no real inviscid flow, it is
just a simplification.

In a viscous flow, the gradients derived from the viscosity effects are so big
that they cannot longer be neglected.

#### Compressible vs Incompressible flow
A flow is said to be Incompressible if no significant changes of the flow's
density are present along the flow section of interest, in other words, the
density of the flow remains constant across all stages of the flow under study,
specially near the zones of interactions with other bodies. On the other hand,
compressible flows have a varying density depending on where the flow is being
analyzed.

The compresibility of a fluid is exremely related to the velocity of sound at a
given condition, which in turn, lets us put the compresibility of a particular
flow in terms of the Mach number. As a general rule one can safely say (at
least for lean bodies, like an airfoil) that a flow is totally incompressible
for mach numbers less than or equal to 0.3. For mach numbers that are less or
equal to 0.8 the flow can still be modeled as incompressible. In the case of
mach numbers between 0.8 and 1.2, the flow is called transonic because there
will be zones in the flow that have a subsonic behaviour whereas there miht be
others in supersonic regime. For flows with mach numbers higher than 1.2, the
flow can safely be said that it behaves under supersonic regime.

- 0.0 < M <= 0.3 -> subsonic
- 0.3 < M <= 0.8 -> still subsonic
- 0.8 < M <= 1.2 -> transonic
- 1.2 < M <= 0.5 -> supersonic
- 0.5 < M        -> hypersonic
