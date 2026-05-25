from Models import Simulation_scheduledmobility

from PIL import Image, ImageDraw

'''
5. ADDITIONAL FILE: MODEL GIF AND OTHER LOCATIONS.
This script creates a dynamic visualization of our scheduled mobility 
model and its environment. By entering any valid Open Street Maps location 
as a string, this code can also run the model in that environment. Just 
change the input value for "location" in the final lines below, or play 
around with other parameter settings.
'''

def hour_of_day(t, step_minutes=5):
    return (t * step_minutes) / 60.0 % 24

def prerender_buildings(sim, scale=1):
    """
    Pre-render buildings once as a static Pillow image.
    scale = 1 means 1 pixel per meter.
    """
    W, H = sim.X, sim.Y

    # White background
    img = Image.new("RGB", (W * scale, H * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw all building pixels once
    for building in sim.building_coords:
      for (x, y) in building:
        draw.point((x * scale, y * scale), fill=(180, 180, 180)) # light gray

    return img

def draw_frame_pillow(sim, t, base_img, scale=1):
    """
    Draw a single frame using Pillow.
    base_img = pre-rendered building background (Pillow Image).
    Draws:
      - agents
      - ALL edges (w >= 1)
      - timestamp text
    """

    # Copy background (fast)
    frame = base_img.copy()
    draw = ImageDraw.Draw(frame)

    # ---- AGENTS: small blue circles ----
    for n in sim.G.nodes:
        x, y = sim.G.nodes[n]["position"]
        draw.ellipse(
            (x*scale-2, y*scale-2, x*scale+2, y*scale+2),
            fill=(0, 0, 255)
        )

    # ---- EDGES: include w=1 ----
    for u, v in sim.G.edges:
        x1, y1 = sim.G.nodes[u]["position"]
        x2, y2 = sim.G.nodes[v]["position"]
        draw.line(
            (x1*scale, y1*scale, x2*scale, y2*scale),
            fill=(0, 0, 0),
            width=1
        )

    # ---- TEXT: timestep + hour ----
    hr = int(round(hour_of_day(t % 288)))
    draw.text((10, 10), f"t={t}, hour={hr:.2f}", fill=(0, 0, 0))

    return frame

# ============================================================
# 8. MAKE GIF
# ============================================================

def make_gif(sim, gif_name='scheduled_mobility.gif', scale=1):
    """
    Much faster GIF renderer:
      - pre-renders building background
      - draws frames using Pillow
      - preserves all edges (including w=1)
    """

    # Pre-render the building layer once
    base_img = prerender_buildings(sim, scale=scale)

    frames = []

    for t in range(sim.T):
        sim.step(t)
        frame_img = draw_frame_pillow(sim, t, base_img, scale=scale)
        frames.append(frame_img)

    # Save the GIF
    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0
    )
    print(f"GIF saved: {gif_name}")

N = 489
T = 288
s = 148
r = 5.98
frac_homes = .3
PW = 0.5
WM = 25
location = "Technical University of Denmark, Lyngby, Denmark"

#Initialize:
sim = Simulation_scheduledmobility(N,T,s,r,PW,WM,frac_homes,location=location)

make_gif(sim,scale=1)

from IPython.display import display, Image as IPImage

display(IPImage(filename='scheduled_mobility.gif'))