from bokeh.models import CustomJS

__all__ = ["arrow_head_marker"]

def arrow_head_marker():
    return CustomJS(code="""
    // Return a Path2D for a chevron-style ">" pointing toward +x at angle = 0
    export default (args, obj, {r}) => {
      // Tunable proportions
      // made by ChatGPT after some adjustments 
      const outer_len   = 2.5 * r;   // tip x from origin
      const inner_len   = 1.0 * r;   // how far the inner notch goes to -x
      const half_height = 1.5 * r;   // half of the overall height

      const path = new Path2D();

      // Concave polygon making a filled chevron ">"
      // bottom outer -> tip -> top outer -> inner notch -> close
      path.moveTo(-inner_len, -half_height);
      path.lineTo( outer_len,  0);
      path.lineTo(-inner_len,  half_height);
      path.lineTo(0, 0);
      path.closePath();

      return path;
    }
    """)
