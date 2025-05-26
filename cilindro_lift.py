import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons, SubplotTool
import cv2
from tkinter import Tk, filedialog
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import scipy.interpolate




def create_mesh(Xrange, Yrange, Xdiv, Ydiv):
    """Crea la malla para la simulación."""
    x = np.linspace(0, Xrange, Xdiv)
    y = np.linspace(0, Yrange, Ydiv)
    X, Y = np.meshgrid(x, y)
    
    print(f"Mesh dimension: {Xdiv}x{Ydiv}")
    print("--------------------")
    
    return x, y, X, Y

def create_cylinder(Xrange, Yrange, R, dx, dy, Xdiv, Ydiv, x, y):
    """Crea el cilindro y determina qué puntos están dentro de él."""
    n = 360
    tetha = np.linspace(0, 2 * np.pi, n)
    center_x, center_y = Xrange/2, Yrange/2
    
    Xcylinder = center_x + R * (Xrange/4) * np.cos(tetha)
    Ycylinder = center_y + R * (Xrange/4) * np.sin(tetha)
    
    Xaranged = np.round(Xcylinder / dx) * dx
    Yaranged = np.round(Ycylinder / dy) * dy
    
    print("Object contour created")
    print("--------------------")
    
    boundary = Path(np.column_stack([Xaranged, Yaranged]))
    
    cylinder = np.zeros((Ydiv, Xdiv), dtype=bool)
    for j in range(Ydiv):
        for i in range(Xdiv):
            if boundary.contains_point((x[i], y[j])):
                cylinder[j, i] = True
    
    print("Cylinder mask created")
    print("--------------------")
    
    return cylinder, Xaranged, Yaranged, Xcylinder, Ycylinder

def create_model(X, Y, dx, dy, Xdiv, Ydiv, x, y):
    """Crea un modelo basado en coordenadas X e Y y determina qué puntos están dentro."""
    
    boundary = Path(np.column_stack([X, Y]))
    
    model = np.zeros((Ydiv, Xdiv), dtype=bool)
    for j in range(Ydiv):
        for i in range(Xdiv):
            if boundary.contains_point((x[i], y[j])):
                model[j, i] = True
    
    print("Object mask created")
    print("--------------------")
    
    return model, X, Y

def Airfoil_model(R, Xrange, Yrange, n_points=500):
    """Genera los puntos para un perfil aerodinámico usando la transformación de Joukowski."""

    center = -0.1 + 0.1j 

    theta = np.linspace(0, 2 * np.pi, n_points)
    z = center + R * np.exp(1j * theta)  # Círculo en el plano complejo

    def joukowski(z):
        return z + 1/z

    z_airfoil = joukowski(z)

    Xpoints = np.real(z_airfoil)
    Ypoints = np.imag(z_airfoil)
    
    x_min, x_max = np.min(Xpoints), np.max(Xpoints)
    y_min, y_max = np.min(Ypoints), np.max(Ypoints)
    
    scale_factor = 0.6 * min(Xrange, Yrange) / max(x_max - x_min, y_max - y_min)
    
    Xpoints = (Xpoints - (x_min + x_max)/2) * scale_factor + Xrange/2
    Ypoints = (Ypoints - (y_min + y_max)/2) * scale_factor + Yrange/2
 
    print("Airfoil geometry created")
    return Xpoints, Ypoints

def process_png_image(image_path, target_width, target_height):
    """Procesa una imagen PNG para extraer su contorno y redimensionarla."""
    print(f"Procesando imagen: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img.shape[2] < 4:
        print("La imagen no tiene canal alfa (transparencia). Usando umbral por brillo.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        alpha = img[:, :, 3]
        _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No se encontraron contornos en la imagen")
        return None, None
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    main_contour = max(contours, key=cv2.contourArea)
    
    X = main_contour[:, 0, 0].astype(np.float32)
    Y = main_contour[:, 0, 1].astype(np.float32)
    
    min_x, max_x = X.min(), X.max()
    min_y, max_y = Y.min(), Y.max()
    
    scale_x = target_width / (max_x - min_x)
    scale_y = target_height / (max_y - min_y)
    scale = min(scale_x, scale_y)
    
    X_scaled = (X - min_x) * scale
    Y_scaled = (Y - min_y) * scale
    
    X_final = X_scaled + (target_width - X_scaled.max()) / 2 + Xrange/4
    Y_final = Y_scaled + (target_height - Y_scaled.max()) / 2
    
    Y_final = -Y_final + target_height + Yrange/4
    
    print("Procesamiento de imagen completado")
    
    return X_final, Y_final

def select_png_file():
    """Abre un diálogo para seleccionar un archivo PNG."""
    root = Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen PNG",
        filetypes=[("Archivos PNG", "*.png"), ("Todos los archivos", "*.*")]
    )
    
    root.destroy()
    
    if file_path:
        print(f"Archivo seleccionado: {file_path}")
        return file_path
    else:
        print("No se seleccionó ningún archivo")
        return None

def initialize_stream_function(Uinf, Xdiv, Ydiv, x, y, model, C, Yrange):
    """Inicializa la función de corriente con condiciones de flujo libre."""
    stream_funcion = np.zeros((Ydiv, Xdiv))
    
    for j in range(Ydiv):
        for i in range(Xdiv):
            if model[j, i]:
                stream_funcion[j, i] = C
            else:
                stream_funcion[j, i] = Uinf * (y[j])
    

    for i in range(Xdiv):
        stream_funcion[0, i] = 0
        stream_funcion[-1, i] = Uinf * Yrange
    
    for j in range(Ydiv):
        stream_funcion[j, 0] = Uinf * y[j]
        stream_funcion[j, -1] = Uinf * y[j]
    
    return stream_funcion

def solve_laplace(sf, model, Xdiv, Ydiv, iterations, tol=0.000001):
    """Resuelve la ecuación de Laplace para la función de corriente."""
    print("Starting numerical solution...")
    
    for iter in range(iterations):
        sf_new = np.copy(sf)
        
        for j in range(1, Ydiv-1):
            for i in range(1, Xdiv-1):
                if not model[j, i]:
                    sf_new[j, i] = 0.25 * (sf[j, i+1] + sf[j, i-1] + sf[j+1, i] + sf[j-1, i])
        max_error = np.max(np.abs(sf_new - sf))
        sf = sf_new
        
        
        if (iter + 1) % (iterations//10) == 0:
            progress = (iter + 1) / iterations * 100
            print(f"{int(progress)}% done")
        
        if max_error < tol:
            print(f"Converged after {iter+1} iterations with error: {max_error:.8e}")
            break
    
        
    print("Solution completed")
    return sf


def calculate_velocity_field(stream_function, dx, dy, Xdiv, Ydiv, model):
    """Calcula el campo de velocidad a partir de la función de corriente."""
    u = np.zeros((Ydiv, Xdiv))
    v = np.zeros((Ydiv, Xdiv))
    
    for j in range(1, Ydiv-1):
        for i in range(1, Xdiv-1):
            if not model[j, i]:
                u[j, i] = (stream_function[j+1, i] - stream_function[j-1, i]) / (2 * dy)
                v[j, i] = -(stream_function[j, i+1] - stream_function[j, i-1]) / (2 * dx)
    for i in range(Xdiv):
        u[0, i] = Uinf
        v[0, i] = 0
        v[-1, i] = 0
        u[-1, i] = Uinf
    
    for j in range(Ydiv):
        u[j, 0] = Uinf
        u[j, -1] = Uinf
        v[j, 0] = 0
        v[j, -1] = 0
    velocity = np.sqrt(u**2 + v**2)
    
    return u, v, velocity

def calculate_pressure_field(Pinf, Uinf, velocity, density):
    """Calcula el campo de presión usando la ecuación de Bernoulli."""

    pressure = Pinf + 0.5*density*Uinf**2 - 0.5*density*velocity**2

 
    return pressure
def change_model(label, simulation_data):
    """Cambia entre modelo de perfil aerodinámico y cilindro."""
    
    fig = simulation_data['fig']
    
    if label == "Otro Modelo":
        png_file = select_png_file()
        if png_file:
            simulation_data['png_file'] = png_file
            X_png, Y_png = process_png_image(png_file, simulation_data['Xrange']/2, simulation_data['Yrange']/2)
            
            if X_png is not None and Y_png is not None:
                simulation_data['custom_model_coords'] = (X_png, Y_png)
            else:
                label = "Perfil Aerodinámico"
        else:
            return
    
    simulation_data['current_model'] = label
    
    fig.canvas.draw_idle()
    
    update(None, simulation_data)

def AoA_change(Xpoints, Ypoints, Xrange, Yrange, alpha):

    center_x, center_y = Xrange/2, Yrange/2
        
    Xpoints_centered = Xpoints - center_x
    Ypoints_centered = Ypoints - center_y
       
    Xpoints_rotated = Xpoints_centered * np.cos(alpha) - Ypoints_centered * np.sin(alpha)
    Ypoints_rotated = Xpoints_centered * np.sin(alpha) + Ypoints_centered * np.cos(alpha)
        
    Xpoints = Xpoints_rotated + center_x
    Ypoints = Ypoints_rotated + center_y
    
    return Xpoints, Ypoints

def calculate_lift(simulation_data):
    """
    Calculates the lift force by integrating the pressure around the model contour.
    
    Args:
        simulation_data: Dictionary containing simulation parameters and results
    
    Returns:
        lift: The calculated lift force
        lift_coefficient: The lift coefficient
    """
    import numpy as np
    
    # Extract necessary data from simulation_data
    X = simulation_data['X']
    Y = simulation_data['Y']
    Xplot = simulation_data['Xplot']
    Yplot = simulation_data['Yplot']
    pressure = simulation_data.get('pressure')
    Uinf = simulation_data['Uinf_slider'].val
    density = simulation_data['density']
    
    if pressure is None:
        # If pressure field hasn't been stored in simulation_data, recalculate it
        dx = simulation_data['dx']
        dy = simulation_data['dy']
        Xdiv = simulation_data['Xdiv']
        Ydiv = simulation_data['Ydiv']
        model = simulation_data['model']
        stream_function = simulation_data.get('stream_function')
        Pinf = simulation_data['Pinf']
        
        if stream_function is None:
            print("Error: Stream function not found in simulation data")
            return 0, 0
            
        # Calculate velocity field
        u, v, velocity = calculate_velocity_field(stream_function, dx, dy, Xdiv, Ydiv, model)
        
        # Calculate pressure field
        pressure = calculate_pressure_field(Pinf, Uinf, velocity, density)
    
    # Interpolate pressure values at model contour points
    
    
    # Reshape grid coordinates for interpolation
    points = np.column_stack((X.flatten(), Y.flatten()))
    values = pressure.flatten()
    
    # Interpolate pressure at the contour points
    contour_pressure = scipy.interpolate.griddata(points, values, (Xplot, Yplot), method='linear')    
    normals_x = np.zeros_like(Xplot)
    normals_y = np.zeros_like(Yplot)
    
    # Calculate normal vectors (simplified approach for closed contour)
    n_points = len(Xplot)
    for i in range(n_points):
        # Use neighboring points to determine tangent direction
        prev_idx = (i - 1) % n_points
        next_idx = (i + 1) % n_points
        
        # Tangent vector
        tangent_x = Xplot[next_idx] - Xplot[prev_idx]
        tangent_y = Yplot[next_idx] - Yplot[prev_idx]
        
        # Normal vector (perpendicular to tangent)
        normals_x[i] = -tangent_y
        normals_y[i] = tangent_x
        
        # Normalize
        magnitude = np.sqrt(normals_x[i]**2 + normals_y[i]**2)
        if magnitude > 0:
            normals_x[i] /= magnitude
            normals_y[i] /= magnitude
    
    # Calculate segment lengths for integration
    segment_lengths = np.zeros(n_points)
    for i in range(n_points):
        next_idx = (i + 1) % n_points
        dx = Xplot[next_idx] - Xplot[i]
        dy = Yplot[next_idx] - Yplot[i]
        segment_lengths[i] = np.sqrt(dx**2 + dy**2)
    
    # Calculate pressure force components
    fx = np.zeros(n_points)
    fy = np.zeros(n_points)
    
    for i in range(n_points):
        if not np.isnan(contour_pressure[i]):
            # Pressure force = pressure * normal vector * segment length
            fx[i] = contour_pressure[i] * normals_x[i] * segment_lengths[i]
            fy[i] = contour_pressure[i] * normals_y[i] * segment_lengths[i]
    
    total_fy = np.sum(fy)
    chord_length = np.max(Xplot) - np.min(Xplot)
    lift_coefficient = total_fy / (0.5 * density * Uinf**2 * chord_length)
    return lift_coefficient

def update(val, simulation_data):
    """Actualiza la visualización cuando se cambia el valor de C, AoA o dx&dy"""
    density = simulation_data['density']

    Yrange = simulation_data['Yrange']
    Xdiv = simulation_data['Xdiv']
    Ydiv = simulation_data['Ydiv']
    x = simulation_data['x']
    y = simulation_data['y']
    X = simulation_data['X']
    Y = simulation_data['Y']
    model = simulation_data['model']
    psi = simulation_data['psi']
    vel = simulation_data['vel']
    pres = simulation_data['pres']
    vel_vector = simulation_data['vel_vector']
    Pinf = simulation_data['Pinf']
    fig = simulation_data['fig']
    c_slider = simulation_data['c_slider']
    iterations = simulation_data['iterations']
    modos = simulation_data['modos']
    
    current_model = simulation_data.get('current_model', modos.value_selected)
    
    Xrange = simulation_data['Xrange']
    R = simulation_data['R']
    dx = simulation_data['dx']
    dy = simulation_data['dy']
    res_slider = simulation_data.get("res_slider", None)
    angle_slider = simulation_data.get("angle_slider", None)
    Uinf_slider = simulation_data.get("Uinf_slider", None)

    C = c_slider.val
    alpha = np.radians(angle_slider.val)
    Uinf = Uinf_slider.val
    Pinf = Pinf_slider.val
    dx = res_slider.val
    dy = res_slider.val

    Xdiv = int(Xrange / dx + 1)
    Ydiv = int(Yrange / dy + 1)
    
    
    
    x, y, X, Y = create_mesh(Xrange, Yrange, Xdiv, Ydiv)

    if current_model == 'Perfil Aerodinámico':
        Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange, n_points=500)
        if angle_slider:
            Xpoints, Ypoints = AoA_change(Xpoints, Ypoints, Xrange, Yrange, -alpha)
        model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
        Xplot, Yplot = Xpoints, Ypoints
    elif current_model == 'Otro Modelo':
        if 'custom_model_coords' in simulation_data:
            Xpoints, Ypoints = simulation_data['custom_model_coords']
            if angle_slider:
                Xpoints, Ypoints = AoA_change(Xpoints, Ypoints, Xrange, Yrange, -alpha)
            model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
            Xplot, Yplot = Xpoints, Ypoints
        else:
            png_file = select_png_file()
            if png_file:
                X_png, Y_png = process_png_image(png_file, Xrange/2, Yrange/2)
                if X_png is not None and Y_png is not None:
                    simulation_data['custom_model_coords'] = (X_png, Y_png)
                    Xpoints, Ypoints = X_png, Y_png
                    model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
                    Xplot, Yplot = Xpoints, Ypoints
                else:
                    Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange, n_points=500)
                    model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
                    Xplot, Yplot = Xpoints, Ypoints
            else:
                Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange, n_points=500)
                model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
                Xplot, Yplot = Xpoints, Ypoints
    else:  
        
        model, Xaranged, Yaranged, Xcylinder_HD, Ycylinder_HD = create_cylinder(Xrange, Yrange, 0.3, dx, dy, Xdiv, Ydiv, x, y)
        Xplot, Yplot = Xcylinder_HD, Ycylinder_HD

    simulation_data['x'] = x
    simulation_data['y'] = y
    simulation_data['X'] = X
    simulation_data['Y'] = Y
    simulation_data['Xdiv'] = Xdiv
    simulation_data['Ydiv'] = Ydiv
    simulation_data['dx'] = dx
    simulation_data['dy'] = dy
    simulation_data["model"] = model
    simulation_data["Xplot"] = Xplot 
    simulation_data["Yplot"] = Yplot
    simulation_data['current_model'] = current_model
    simulation_data['iterations'] = iterations
    
    stream = initialize_stream_function(Uinf, Xdiv, Ydiv, x, y, model, C, Yrange)

    stream = solve_laplace(stream, model, Xdiv, Ydiv, iterations)
    
    # Store stream function for later use in lift calculation
    simulation_data['stream_function'] = stream

    u, v, velocity = calculate_velocity_field(stream, dx, dy, Xdiv, Ydiv, model)
    
    pressure = calculate_pressure_field(Pinf, Uinf, velocity, density)
    
    # Store pressure field for later use in lift calculation
    simulation_data['pressure'] = pressure
    
    # Calculate lift and lift coefficient
    lift = calculate_lift(simulation_data)
    
    # Store lift values
    simulation_data['lift'] = lift
 
    
    psi.clear()
    vel.clear()
    pres.clear()
    vel_vector.clear()

    psi.set_title("Streamfunction")
    vel.set_title("Velocitat")
    pres.set_title("Pressió")
    vel_vector.set_title("Vector Velocitat")

    psi.set_xlim(0, Xrange)
    psi.set_ylim(0, Yrange)
    vel.set_xlim(0, Xrange)
    vel.set_ylim(0, Yrange)
    pres.set_xlim(0, Xrange)
    pres.set_ylim(0, Yrange)
    vel_vector.set_ylim(0, Yrange)
    vel_vector.set_ylim(0, Yrange)

    C = c_slider.val
    stream_range = Uinf * Yrange
    stream_levels = np.linspace(-stream_range*0.1, stream_range*1.1, 25)
    


    graph_stream = psi.contourf(X, Y, stream, levels=stream_levels, cmap='inferno', extend='both', alpha=0.8)
    psi.contour(X, Y, stream, levels=stream_levels, colors='black', linewidths=0.5, alpha=0.6)
    psi.set_aspect('equal')
    
    
    graph_vel = vel.pcolormesh(X, Y, velocity, cmap='PuBuGn', shading='gouraud')
    vel.set_aspect('equal')
    
    
    graph_pres = pres.pcolormesh(X, Y, pressure, cmap='YlOrRd', shading='gouraud')
    pres.set_aspect('equal')
    

    skip =10
    mask = ~model[::skip, ::skip]
    velocidad_grafo = np.sqrt(u[::skip, ::skip]**2 + v[::skip, ::skip]**2)
    max_vel = np.max(velocidad_grafo[mask])
    scale = max_vel * 3
    
    graph_vel_vector = vel_vector.quiver(X[::skip, ::skip], Y[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], velocidad_grafo, color='black', scale=3000, width=0.002, headwidth=4, headlength=5, headaxislength=4.5, pivot='mid', cmap='PuBuGn')
    
    title = f"Streamlines - {current_model} (C = {C:.2f})"
    if angle_slider and current_model == 'Perfil Aerodinámico':
        title += f", Ángulo = {angle_slider.val}°"
    
    psi.set_title(title)
    
    psi.plot(Xplot, Yplot, linewidth=2, color='black')
    vel.plot(Xplot, Yplot, linewidth=2, color = "white")
    pres.plot(Xplot, Yplot, linewidth=2, color='black')
    vel_vector.plot(Xplot, Yplot, linewidth=2, color='black')
    psi.fill(Xplot, Yplot, alpha=1, color='black')
    vel.fill(Xplot, Yplot, alpha=1, color='white')
    pres.fill(Xplot, Yplot, alpha=1, color='black')
    vel_vector.fill(Xplot, Yplot, alpha=1, color='black')

    if 'cbar_stream' in simulation_data:
        try:
            simulation_data['cbar_stream'].remove()
        except (AttributeError, ValueError):
            pass
        simulation_data['cbar_stream'] = None
        
    if 'cbar_vel' in simulation_data:
        try:
            simulation_data['cbar_vel'].remove()
        except (AttributeError, ValueError):
            pass
        simulation_data['cbar_vel'] = None
        
    if 'cbar_pres' in simulation_data:
        try:
            simulation_data['cbar_pres'].remove()
        except (AttributeError, ValueError):
            pass
        simulation_data['cbar_pres'] = None
    
    streamBar = make_axes_locatable(psi)
    cax_stream = streamBar.append_axes("right", size="5%", pad=0.05)
    cbar_stream = fig.colorbar(graph_stream, cax=cax_stream)
    simulation_data['cbar_stream'] = cbar_stream
    
    velBar = make_axes_locatable(vel)
    cax_vel = velBar.append_axes("right", size="5%", pad=0.05)
    cbar_vel = fig.colorbar(graph_vel, cax=cax_vel)
    cbar_vel.set_label('m/s')
    simulation_data['cbar_vel'] = cbar_vel
    
    presBar = make_axes_locatable(pres)
    cax_pres = presBar.append_axes("right", size="5%", pad=0.05)
    cbar_pres = fig.colorbar(graph_pres, cax=cax_pres)
    cbar_pres.set_label('Pa')
    simulation_data['cbar_pres'] = cbar_pres

    psi.text(0.02, 0.02, f"Resolución: {dx:.4f}\nIteraciones: {iterations}",
            transform=psi.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add lift coefficient information to the visualization
    vel.text(0.02, 0.02, f"Lift: {lift:.2f} N\nLift: {lift:.3f}",
            transform=vel.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.canvas.draw_idle()
    print("Visualización actualizada")
    print(f"Lift: {lift:.2f} N, Lift: {lift:.3f}")

Uinf = 60            # Velocidad del flujo libre
Pinf = 100000
R = 1.1              # Radio de referencia
dx = 0.05            # Paso en X (resolución moderada para velocidad)
dy = 0.05            # Paso en Y
Xrange = 2 * R       # Rango en X
Yrange = 1 * R       # Rango en Y
density = 1.225      # Densidad
iterations = 20000
    
Xdiv = int(Xrange / dx + 1)
Ydiv = int(Yrange / dy + 1)
 
print("Configurando simulación...")
print(f"Speed = {Uinf}")
print(f"Domain dimensions: {Xrange}x{Yrange}")
print(f"Resolution: dx={dx}, dy={dy}")
print("--------------------")
    
x, y, X, Y = create_mesh(Xrange, Yrange, Xdiv, Ydiv)
  
Xpoints, Ypoints = Airfoil_model(R, Xrange, Yrange)
model, Xaranged, Yaranged = create_model(Xpoints, Ypoints, dx, dy, Xdiv, Ydiv, x, y)
   
C_default = (Uinf * Yrange) * 0.5

fig = plt.figure(figsize=(18,8)) 

gs = GridSpec(2, 2, figure=fig, hspace= 0.17, wspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.15)

psi = fig.add_subplot(gs[1, 0])
vel = fig.add_subplot(gs[0, 0])
pres = fig.add_subplot(gs[0, 1])
vel_vector = fig.add_subplot(gs[1, 1])

plt.subplots_adjust(bottom=0.5)

ax_c_slider = plt.axes([0.55, 0.04, 0.40, 0.03])
  
c_slider = Slider(ax_c_slider, 'C', 0.0, Uinf*Yrange, valinit=Uinf*Yrange*0.5, valstep=1)
    
ax_angle_slider = plt.axes([0.55, 0.02, 0.40, 0.03])
    
angle_slider = Slider(ax_angle_slider, 'Ángulo de Ataque (°)', -15, 15, valinit=0, valstep=1)

ax_res_slider = plt.axes([0.55, 0, 0.40, 0.03])

res_slider = Slider(ax_res_slider, "Resolución de malla", 0.001, 0.1, valinit=0.1, valstep=0.001)

ax_Uinf_slider = plt.axes([0.55, 0.06, 0.40, 0.03])

Uinf_slider = Slider(ax_Uinf_slider, "Uinf", 5, 100, valinit=60, valstep=5)

ax_Pinf_slider = plt.axes([0.55, 0.08, 0.40, 0.03])

Pinf_slider = Slider(ax_Pinf_slider, "Pinf", 20000, 200000, valinit=100000, valstep=500)
   
ax_modos = plt.axes([0, 0, 0.3, 0.05])
modos = RadioButtons(ax_modos, ('Perfil Aerodinámico', 'Cilindro', "Otro Modelo"), active=0)

simulation_data ={
    'Uinf': Uinf,
    'Yrange': Yrange,
    'Xdiv': Xdiv,
    'Ydiv': Ydiv,
    'x': x,
    'y': y,
    'X': X,
    'Y': Y,
    'model': model,
    'Xaranged': Xaranged,
    'Yaranged': Yaranged,
    'Xplot': Xpoints, 
    'Yplot': Ypoints,
    'psi': psi,
    'vel': vel,
    'pres': pres,
    'fig': fig,
    'c_slider': c_slider,
    'angle_slider': angle_slider,
    "res_slider" : res_slider,
    "Uinf_slider" : Uinf_slider,
    "Pinf_slider" : Pinf_slider,
    "vel_vector" : vel_vector,
    'iterations': iterations,
    'modos': modos,
    'dx': dx,
    'dy': dy,
    'R': R,
    'Xrange': Xrange,
    'cbar': None,
    'Pinf': Pinf,
    'density': density,
    'current_model': 'Perfil Aerodinámico'
}
c_slider.on_changed(lambda val: update(val, simulation_data))
angle_slider.on_changed(lambda val: update(val, simulation_data))
res_slider.on_changed(lambda val: update(val, simulation_data))
Uinf_slider.on_changed(lambda val: update(val, simulation_data))
Pinf_slider.on_changed(lambda val: update(val, simulation_data))
    
modos.on_clicked(lambda label: change_model(label, simulation_data))
    
update(None, simulation_data)


print("Simulación lista. Ajusta el modelo o los sliders para ver cambios automáticos.")
plt.show()
