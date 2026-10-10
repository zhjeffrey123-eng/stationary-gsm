"""按论文保存的真实矢量几何复绘四幅主图，不插值、不平滑、不改数据。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Polygon
from matplotlib.font_manager import findfont,FontProperties
from .data import PAPER


def render(root,target):
    target = Path(target)
    target.mkdir(parents=True,exist_ok=False)
    # 要求目标字体存在，不把其他字体默认为 Times New Roman。
    findfont(FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    counts = []
    for name in ('Figure_1','Figure_2','Figure_3','Figure_4'):
        geo = json.loads((Path(root)/PAPER/'figures'/f'{name}.json').read_text(encoding='utf-8'))
        fig = plt.figure(figsize=(geo['width'],geo['height']),facecolor='white')
        ax = fig.add_axes([0,0,1,1],xlim=(0,geo['width']),ylim=(0,geo['height']))
        ax.set_axis_off()
        for j,s in enumerate(geo['shapes']):
            color = np.array([int(v) for v in s['color'].split(',')])/255
            style = {0:'none',1:'-',2:'--',3:':',4:'-.'}[s.get('pattern',1)]
            weight = s.get('weight',.65) or .65
            zorder = j/10000+1
            if s['kind'] in ('line','polyline'):
                points = np.array(s['points']) if s['kind']=='polyline' else np.array([[s['x'],s['y']],[s['xx'],s['yy']]])
                ax.plot(points[:,0],points[:,1],color=color,linewidth=weight,linestyle=style,zorder=zorder)
                if s.get('arrow'):
                    direction = points[-1]-points[-2]
                    direction /= np.linalg.norm(direction)
                    normal = np.array([-direction[1],direction[0]])
                    rear = points[-1]-.055*direction
                    ax.add_patch(Polygon([points[-1],rear+.025*normal,rear-.025*normal],
                        facecolor=color,edgecolor='none',zorder=zorder))
            elif s['kind'] in ('rect','heatcell'):
                face = color if s.get('fill') or s['kind']=='heatcell' else 'white'
                if s.get('transparent'):
                    face = 'none'
                ax.add_patch(Rectangle((s['x'],s['y']),s['xx']-s['x'],s['yy']-s['y'],
                    facecolor=face,edgecolor='none' if s['kind']=='heatcell' else color,
                    linewidth=weight,linestyle=style,zorder=zorder))
            elif s['kind']=='marker':
                x,y,r = s['x'],s['y'],s['size']
                marker = s['marker']
                face = color if s.get('fill') else 'white'
                if marker=='circle':
                    patch = Circle((x,y),r,facecolor=face,edgecolor=color,linewidth=.7,zorder=zorder)
                elif marker=='square':
                    patch = Rectangle((x-r,y-r),2*r,2*r,facecolor=face,edgecolor=color,linewidth=.7,zorder=zorder)
                elif marker=='triangle':
                    patch = Polygon([[x,y+r],[x-r,y-r],[x+r,y-r]],facecolor='none',edgecolor=color,linewidth=.7,zorder=zorder)
                else:
                    ax.plot([x-r,x+r],[y-r,y+r],color=color,linewidth=.7,zorder=zorder)
                    ax.plot([x-r,x+r],[y+r,y-r],color=color,linewidth=.7,zorder=zorder)
                    continue
                ax.add_patch(patch)
            elif s['kind']=='text':
                ax.text(s['x'],s['y'],s['text'],ha='center',va='center',rotation=s['angle'],
                    fontsize=8,fontweight='bold' if s.get('bold') else 'normal',color=color,
                    parse_math=False,zorder=zorder)
            else:
                raise ValueError('不支持的几何对象 '+s['kind'])
        for suffix in ('pdf','svg','png'):
            # 不用 tight 裁切，保持实际单栏尺寸和字号。
            fig.savefig(target/f'{name}.{suffix}',dpi=300)
        plt.close(fig)
        counts.append(len(geo['shapes']))
    return {'figures':4,'shape_counts':counts,'font':'Times New Roman','font_size_pt':8,
            'outputs':['PDF','SVG','PNG'],'interpolation':'none'}
