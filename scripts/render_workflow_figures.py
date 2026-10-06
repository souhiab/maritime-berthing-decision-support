"""Render source-grounded workflow figures. Run from any working directory.

Uses the project's existing Matplotlib dependency. No model execution, downloads,
API keys, or changes to research outputs are required.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
BG, INK, MUTED, LINE = "#FBFAF7", "#152A3B", "#526574", "#D9E1E5"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "savefig.facecolor": BG})
TEXT_CHECKS = []

def text(ax, x, y, s, size=11.5, color=INK, weight="normal", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   va="top", linespacing=1.45, zorder=5, **kw)

def panel(ax, x, y, w, h, fc="white", ec=LINE, dashed=False, radius=.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fc,
        edgecolor=ec, linewidth=1.1, linestyle="--" if dashed else "-", zorder=2))

def draw(figspec, spec):
    fig = plt.figure(figsize=(16,10), facecolor=BG)
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,100),ylim=(62.5,0)); ax.axis("off")
    accent = spec["accent"]
    ax.plot([5,8],[3.15,3.15],lw=3.5,color=accent,solid_capstyle="round")
    text(ax,9,2.65,spec["name"].upper(),10,accent,"bold")
    text(ax,95,2.65,f'{figspec["index"]:02d} / {figspec["level"].upper()}',10,MUTED,ha="right")
    text(ax,5,6,figspec["title"],29,INK,"bold")
    text(ax,5,10.6,figspec["subtitle"],12,MUTED)
    ax.plot([5,95],[14.1,14.1],lw=.8,color=LINE)
    for p in figspec.get("panels",[]):
        panel(ax,*p["rect"],fc=p.get("fill","#F1F4F5"),ec=p.get("border",LINE),dashed=p.get("dashed",False))
        if p.get("label"):text(ax,p["rect"][0]+1.4,p["rect"][1]+1,p["label"],9.5,MUTED,"bold")
    # Draw routed connections behind cards. Solid = implemented, dashed = proposed.
    for e in figspec.get("edges",[]):
        points=e["points"]; color=e.get("color",accent)
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.2,
                    linestyle="--" if e.get("dashed") else "-",zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",
            mutation_scale=12,linewidth=1.2,color=color,
            linestyle="--" if e.get("dashed") else "-",zorder=1))
        if "label" in e:
            text(ax,*e["at"],e["label"],9.5,MUTED,
                 bbox=dict(facecolor=BG,edgecolor="none",pad=2),ha=e.get("align","center"))
    for c in figspec["cards"]:
        x,y,w,h=c["rect"]; dark=c.get("dark",False); proposed=c.get("proposed",False)
        panel(ax,x,y,w,h,fc=INK if dark else c.get("fill","white"),
              ec=INK if dark else LINE,dashed=proposed)
        tcolor="white" if dark else INK
        eyebrow=text(ax,x+1.4,y+1.1,c["eyebrow"].upper(),9,"#B7DEE1" if dark else accent,"bold")
        title=text(ax,x+1.4,y+3,c["title"],c.get("title_size",15.5),tcolor,"bold")
        title_lines=c["title"].count("\n")+1
        body_y=y+3+(2.5 if title_lines==1 else 4.0)
        body=text(ax,x+1.4,body_y,c["body"],c.get("body_size",11.3),"#E0EAEE" if dark else MUTED)
        TEXT_CHECKS.append((fig,ax,(x+.6,y+.5,w-1.2,h-1),[eyebrow,title,body],c["eyebrow"]))
    for l in figspec.get("labels",[]):
        text(ax,l["x"],l["y"],l["text"],l.get("size",10),l.get("color",MUTED),l.get("weight","normal"),ha=l.get("align","left"))
    # Bottom evidence band keeps measurements apart from the workflow itself.
    panel(ax,5,50.1,90,7.4,fc=spec["tint"],ec=spec["tint"])
    for i,m in enumerate(figspec["evidence"]):
        x=6.7+i*30
        if i:ax.plot([x-1.7,x-1.7],[51.5,56.2],color=LINE,lw=1)
        text(ax,x,51.1,m[0],17.5,INK,"bold")
        text(ax,x,54,m[1],10.2,MUTED)
    text(ax,5,59,figspec["footer"],9.4,MUTED)
    text(ax,95,59,spec["short"]+f' / {figspec["index"]:02d}',9.4,MUTED,ha="right")
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for f,a,rect,artists,label in TEXT_CHECKS:
        if f is not fig: continue
        x,y,w,h=rect
        for artist in artists:
            box=artist.get_window_extent(renderer).transformed(ax.transData.inverted())
            # Inverted y-axis: compare normalized extrema.
            lo,hi=sorted([box.y0,box.y1])
            assert box.x0>=x-.15 and box.x1<=x+w+.15 and lo>=y-.15 and hi<=y+h+.15, (label,artist.get_text(),box.bounds,rect)
    return fig

def main():
    spec=json.loads((ROOT/"docs"/"workflow_figures.json").read_text())
    dest=ROOT/"assets"/"workflows"; dest.mkdir(parents=True,exist_ok=True)
    with PdfPages(dest/"workflow_figures.pdf",metadata={"Title":spec["name"]+" | Visual walkthrough","Author":"Souhaib Benbouazza","Subject":"Source-grounded synthetic portfolio workflow"}) as pdf:
        for fs in spec["figures"]:
            fig=draw(fs,spec)
            fig.savefig(dest/(fs["slug"]+".png"),dpi=140)
            fig.savefig(dest/(fs["slug"]+".svg"))
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Rendered {len(spec["figures"])} figures (PNG, SVG, PDF) into {dest}')

if __name__=="__main__":main()
