# shades

Project on finding the best stadium seats with shade:

Plan:
- Using NOAA's calculator API perhaps
- Need solar position converting to a unit vector, with 
    x = east, 
    y = north, 
    z = up. 
    Making α elevation and γ azimuth measured clockwise from north, we can have the unit vector be
    \vec{s} = (sinγ \cdot cosα, cosγ \cdot cosα, sinα)

- Parametric Bowl: Model the stadium as a set of surfaces of revolution around an ellipse:
Field: ellipse
- According to me and my friends faulty maths, The tilt is $ arctan⁡ ⁣((z2−z1)/(r2−r1)) $
- Google Earth Pro and honestly Wikipedia pages we can get Stadium dimensions
- Some intuition: a section is shaded either because the sun is behind the stands opposite you (late in the game) or because the deck overhead cuts it off. Early-to-mid afternoon it's almost entirely the second one.
- ChatGPT sites: OpenStreetMap — stadiums often have building:part, height, min_height, roof:shape, roof:height tags. Pull with Overpass API or osmnx (ox.features_from_place(name, tags={'building': True})). Extrude the footprints. Quality is wildly variable — check the specific venue before committing. Also configure LiDAR / DSM.
- blender-osm or actaully build it out in blender LMAO 

- Shadow projection: Project the roof/deck edge along −\vec{s} onto the seat in where we get a polygonal shape, and using Shapely library we can point-in-polygon
- For a casting vertex R and a receiving plane through Q_0​ with normal (orthogonal) \vec{n} we have:  
    $$ t=\frac{(Q_0​−R)⋅n}{−s⋅n} , R′=R−t\vec{s} $$
Total solar energy hitting a seat splits into:
    - DNI: direct normal irradiance, the beam from the sun's disk. blocked by shade. ~800–1000 W/m² on a clear summer day.
    - DHI: diffuse horizontal irradiance, light scattered by the whole sky dome. not blocked by being in shadow; only by how much sky you can see. ~100–150 W/m².
    - Reflected: off the field, concrete, seatbacks. Small but nonzero.
- GPT: $$ score(seat)=\sigma_t​[1[sunlit]⋅DNI_t​cos\theta t​+SVF \cdot DHI_t​], $$ where \theta_t​ is the angle between the sun and the seat's facing direction (a seat looking down at the field faces mostly up, so it catches a lot of overhead sun).
- matplotlib top-down scatter of seats colored by shade fraction.