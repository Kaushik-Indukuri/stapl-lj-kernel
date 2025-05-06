#include <algorithm>
#include <benchmarks/algorithms/timer.hpp>
#include <benchmarks/algorithms/utilities.hpp>
#include <chrono>
#include <cmath>
#include <float.h>
#include <iostream>
#include <stapl/array.hpp>
#include <stapl/domains/indexed.hpp>
#include <stapl/numeric.hpp>
#include <stapl/utility/do_once.hpp>
#include <stapl/vector.hpp>
#include <tuple>

using namespace std;

// Define data types
typedef struct {
  double x, y, z;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & x & y & z;
  }
} particle_position_t;

typedef struct {
  double x, y, z;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & x & y & z;
  }
} force_vector_t;

typedef struct {
  double x, y, z;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & x & y & z;
  }
} particle_velocity_t;


// Apply periodic boundary conditions
void apply_pbc(particle_position_t& pos, double dim_x, double dim_y, double dim_z) {
  pos.x = fmod(pos.x + dim_x, dim_x);
  pos.y = fmod(pos.y + dim_y, dim_y);
  pos.z = fmod(pos.z + dim_z, dim_z);
}


// Functor for calculating forces in parallel
struct force_calculation_wf {
  double m_sigma6;
  double m_sigma12;
  double m_epsilon;
  double m_cutoff_sq;
  double m_dim_x, m_dim_y, m_dim_z;
  
  force_calculation_wf(double sigma6, double sigma12, double epsilon, double cutoff_sq,
                      double dim_x, double dim_y, double dim_z)
      : m_sigma6(sigma6), m_sigma12(sigma12), m_epsilon(epsilon), m_cutoff_sq(cutoff_sq),
        m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {}
  
  template <typename P, typename F>
  void operator()(P &&positions_view, F &&forces_output) {
    auto dom = positions_view.domain();   // yields the local index range
    auto first = dom.first();
    auto last  = dom.last();
    auto n = dom.size();

    vector<particle_position_t> local_pos(n);
    for (auto i = first; i <= last; i++)
      local_pos[i - first] = positions_view[i];

    // Fill the forces output
    vector<force_vector_t> forces(n, {0.0, 0.0, 0.0});
    
    // Calculate all pairwise forces
    for (size_t i = 0; i < n; i++) {
      particle_position_t &p1 = local_pos[i];
      for (size_t j = i+1; j < n; j++) {
        particle_position_t &p2 = local_pos[j];

        // Calculate distance vector
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dz = p1.z - p2.z;

        // Apply minimum image convention
        dx -= m_dim_x * round(dx / m_dim_x);
        dy -= m_dim_y * round(dy / m_dim_y);
        dz -= m_dim_z * round(dz / m_dim_z);

        // Calculate squared distance
        double r_squared = dx*dx + dy*dy + dz*dz;

        // Apply force only if particles are within cutoff
        if (r_squared <= 0.0 || r_squared >= m_cutoff_sq) continue; 

        double inv_r2 = 1.0 / r_squared;
        double inv_r6 = inv_r2 * inv_r2 * inv_r2;
        double inv_r12 = inv_r6 * inv_r6;

        // LJ force magnitude: 24ε[(2σ¹²/r¹²) - (σ⁶/r⁶)]/r²
        double force_magnitude = 24.0 * m_epsilon * 
        (2.0 * m_sigma12 * inv_r12 - 
        m_sigma6 * inv_r6) * inv_r2;

        // Apply direction
        double fx = force_magnitude * dx;
        double fy = force_magnitude * dy;
        double fz = force_magnitude * dz;
        
        // Apply Newton's third law: equal and opposite forces
        forces[i].x += fx;
        forces[i].y += fy;
        forces[i].z += fz;
        
        forces[j].x -= fx;
        forces[j].y -= fy;
        forces[j].z -= fz;
      }
    }

    // Copy forces to the output
    for (auto i = first; i <= last; i++) {
      forces_output[i] = forces[i-first];
    }
  }
  
  void define_type(stapl::typer &t) {
    t.member(m_sigma6);
    t.member(m_sigma12);
    t.member(m_epsilon);
    t.member(m_cutoff_sq);
    t.member(m_dim_x);
    t.member(m_dim_y);
    t.member(m_dim_z);
  }
};


// Functor for Velocity-Verlet integration
struct velocity_verlet_wf {
  double m_timestep;
  double m_mass;
  double m_dim_x, m_dim_y, m_dim_z;
  vector<particle_velocity_t> m_velocities;
  
  velocity_verlet_wf(double timestep, double mass, vector<particle_velocity_t> &velocities, 
                     double dim_x, double dim_y, double dim_z)
    : m_timestep(timestep), m_mass(mass), m_velocities(velocities), 
      m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {}
  
  template <typename P, typename F, typename O>
  void operator()(P &&positions_view, F &&forces_view, O &&new_positions_output) {
    auto dom = positions_view.domain();   // yields the local index range
    auto first = dom.first();
    auto last  = dom.last();
    
    // Update positions and velocities using velocity-Verlet algorithm
    for (auto i = first; i <= last; i++) {
      // Get the base data through the proxy
      particle_position_t pos_val = static_cast<particle_position_t>(positions_view[i]);
      force_vector_t force_val = static_cast<force_vector_t>(forces_view[i]);
      
      // Position update
      particle_position_t new_position;
      new_position.x = pos_val.x + m_velocities[i].x * m_timestep + 
                          0.5 * force_val.x / m_mass * m_timestep * m_timestep;
      new_position.y = pos_val.y + m_velocities[i].y * m_timestep + 
                          0.5 * force_val.y / m_mass * m_timestep * m_timestep;
      new_position.z = pos_val.z + m_velocities[i].z * m_timestep + 
                          0.5 * force_val.z / m_mass * m_timestep * m_timestep;
      
      // Apply periodic boundary conditions
      apply_pbc(new_position, m_dim_x, m_dim_y, m_dim_z);

      // Update the output
      new_positions_output[i] = new_position;

      // Velocity half-update
      m_velocities[i].x += 0.5 * force_val.x / m_mass * m_timestep;
      m_velocities[i].y += 0.5 * force_val.y / m_mass * m_timestep;
      m_velocities[i].z += 0.5 * force_val.z / m_mass * m_timestep;
    }
  }
  
  void define_type(stapl::typer &t) {
    t.member(m_timestep);
    t.member(m_mass);
    t.member(m_velocities);
    t.member(m_dim_x);
    t.member(m_dim_y);
    t.member(m_dim_z);
  }
};


// Functor for computing system energy
struct energy_calculation_wf {
  double m_sigma;
  double m_epsilon;
  double m_cutoff;
  double m_mass;
  double m_dim_x, m_dim_y, m_dim_z;
  vector<particle_velocity_t> m_velocities;
  
  energy_calculation_wf(double sigma, double epsilon, double cutoff, 
                       double mass, vector<particle_velocity_t> &velocities, 
                       double dim_x, double dim_y, double dim_z)
    : m_sigma(sigma), m_epsilon(epsilon), m_cutoff(cutoff), 
      m_mass(mass), m_velocities(velocities),
      m_dim_x(dim_x), m_dim_y(dim_y), m_dim_z(dim_z) {}
  
  template <typename T>
  double operator()(T &&positions_view) {
    auto dom   = positions_view.domain();
    auto first = dom.first();
    auto last  = dom.last();

    double potential_energy = 0.0;
    double kinetic_energy = 0.0;
    double cutoff_sq = m_cutoff * m_cutoff;
    
    
    // Calculate potential energy (LJ potential)
    for (auto i = first; i <= last; i++) {
      particle_position_t pos_i = positions_view[i];
      for (auto j = i+1; j<=last; j++) {
        particle_position_t pos_j = positions_view[j];

        double dx = pos_i.x - pos_j.x;
        double dy = pos_i.y - pos_j.y;
        double dz = pos_i.z - pos_j.z;

        // Apply minimum image convention
        dx -= m_dim_x * round(dx / m_dim_x);
        dy -= m_dim_y * round(dy / m_dim_y);
        dz -= m_dim_z * round(dz / m_dim_z);
        
        double r_squared = dx*dx + dy*dy + dz*dz;
        
        if (r_squared < cutoff_sq && r_squared > 0.0) {
          double r_sq_inv = 1.0 / r_squared;
          double r_6_inv = r_sq_inv * r_sq_inv * r_sq_inv;
          double r_12_inv = r_6_inv * r_6_inv;
          double sigma_6 = m_sigma * m_sigma * m_sigma * m_sigma * m_sigma * m_sigma;
          double sigma_12 = sigma_6 * sigma_6;
          
          // LJ potential: 4ε[(σ/r)¹² - (σ/r)⁶]
          potential_energy += 4.0 * m_epsilon * 
                             (sigma_12 * r_12_inv - 
                              sigma_6 * r_6_inv);
        }
      }
      
      // Calculate kinetic energy
      double v_squared = m_velocities[i].x * m_velocities[i].x + 
                        m_velocities[i].y * m_velocities[i].y + 
                        m_velocities[i].z * m_velocities[i].z;
      kinetic_energy += 0.5 * m_mass * v_squared;
    }
    
    return potential_energy + kinetic_energy;
  }
  
  void define_type(stapl::typer &t) {
    t.member(m_sigma);
    t.member(m_epsilon);
    t.member(m_cutoff);
    t.member(m_mass);
    t.member(m_velocities);
    t.member(m_dim_x);
    t.member(m_dim_y);
    t.member(m_dim_z);
  }
};


void initialize_particles(
  vector<particle_position_t>& positions,
  double dim_x, double dim_y, double dim_z
) {
  int num_particles = positions.size();

  // Calculate cubic dimensions to fit all particles within the box
  int particles_per_side = std::ceil(std::cbrt(num_particles));
  float spacing_x = dim_x / particles_per_side;
  float spacing_y = dim_y / particles_per_side;
  float spacing_z = dim_z / particles_per_side;
  
  // Place particles in a cubic lattice
  int count = 0;
  for (int x = 0; x < particles_per_side && count < num_particles; x++) {
      for (int y = 0; y < particles_per_side && count < num_particles; y++) {
          for (int z = 0; z < particles_per_side && count < num_particles; z++) {
              // Center the lattice in the box and add small offset to avoid particles at exact box edges
              positions[count].x = (x + 0.5) * spacing_x;
              positions[count].y = (y + 0.5) * spacing_y;
              positions[count].z = (z + 0.5) * spacing_z;
              count++;
          }
      }
  }
  
  // Add small random displacements to avoid perfect symmetry
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-0.01 * spacing_x, 0.01 * spacing_x);
  
  for (int i = 0; i < num_particles; i++) {
      positions[i].x += dist(gen);
      positions[i].y += dist(gen);
      positions[i].z += dist(gen);
      
      // Ensure particles are within box boundaries
      positions[i].x = fmod(positions[i].x + dim_x, dim_x);
      positions[i].y = fmod(positions[i].y + dim_y, dim_y);
      positions[i].z = fmod(positions[i].z + dim_z, dim_z);
  }
}


// Initialize velocities for each particle
void initialize_velocities(vector<particle_velocity_t>& velocities, double temperature) {
  int num_particles = velocities.size();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0.0, sqrt(temperature));
  
  for (int i = 0; i < num_particles; i++) {
    velocities[i].x = dist(gen);
    velocities[i].y = dist(gen);
    velocities[i].z = dist(gen);
  }
  
  // Remove center of mass motion
  double cm_vx = 0.0, cm_vy = 0.0, cm_vz = 0.0;
  for (int i = 0; i < num_particles; i++) {
    cm_vx += velocities[i].x;
    cm_vy += velocities[i].y;
    cm_vz += velocities[i].z;
  }
  
  cm_vx /= num_particles;
  cm_vy /= num_particles;
  cm_vz /= num_particles;
  
  for (int i = 0; i < num_particles; i++) {
    velocities[i].x -= cm_vx;
    velocities[i].y -= cm_vy;
    velocities[i].z -= cm_vz;
  }
}


/**
 * @brief Main function for the Lennard-Jones STAPL application.
 *
 * This function performs a Lennard-Jones molecular dynamics simulation to model 
 * a system of particles (atoms or molecules) that interact with each other through 
 * the Lennard-Jones potential. It initializes particles into a simple cubic lattice 
 * and measures the execution time.
 *
 * @details
 * Initializes particles with a fixed mass arranged in a simple cubic lattice. 
 * Simulates their motion over a specified number of timesteps using the Lennard-Jones 
 * potential to model inter-particle interactions. The total execution time of the 
 * simulation is measured and reported.
 * 
 */
stapl::exit_code stapl_main(int argc, char *argv[]) {

  if (argc < 2) {
    stapl::do_once([] {
      std::cout << "usage: ./lj num_particles num_timesteps" << std::endl;
    });
    return EXIT_FAILURE;
  }

  size_t num_particles = 5000;
  size_t num_timesteps = 10000;

  if (argc == 2) {
    num_particles = atol(argv[1]);
  } else if (argc == 3) {
    num_particles = atol(argv[1]);
    num_timesteps = atol(argv[2]);
  }

  counter_t timer;

  // setting up simulation parameters
  const double dim_x = 50.0;
  const double dim_y = 50.0;
  const double dim_z = 50.0;
  const double epsilon = 1.0;
  const double sigma = 1.0;
  const double cutoff = 2.5 * sigma;
  const double timestep = 0.005;
  const double mass = 1.0;
  double temperature = 1.0;  // Initial temperature (in reduced units)

  // Initialize particle positions and velocities
  vector<particle_position_t> positions(num_particles);
  vector<particle_velocity_t> velocities(num_particles);

  initialize_particles(positions, dim_x, dim_y, dim_z);
  initialize_velocities(velocities, temperature);

  // Create STAPL containers and views
  using pos_cont_t = stapl::array<particle_position_t>;
  using pos_view_t = stapl::array_view<pos_cont_t>;
  pos_cont_t positions_array(num_particles);
  pos_view_t positions_view(positions_array);
  pos_cont_t new_positions_array(num_particles);
  pos_view_t new_positions_view(new_positions_array);

  stapl::array<force_vector_t> forces_array(num_particles);
  stapl::array_view<stapl::array<force_vector_t>> forces_view(forces_array);
  
  // Copy initial positions to STAPL container
  for (size_t i = 0; i < num_particles; i++) {
    positions_array[i] = positions[i];
  }
  
  std::vector<double> execution_times(num_timesteps, 0.0);
  double sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
  double sigma12 = sigma6 * sigma6;
  double cutoff_sq = cutoff * cutoff;
  
  // Main simulation loop
  timer.reset();
  timer.start();
  for (size_t step = 0; step < num_timesteps; step++) {
    // Calculate forces
    stapl::map_func<stapl::skeletons::tags::with_coarsened_wf>(
            force_calculation_wf(sigma6, sigma12, epsilon, cutoff_sq, dim_x, dim_y, dim_z),
            positions_view,
            forces_view);
    
    // Update positions and half-step velocities
    stapl::map_func<stapl::skeletons::tags::with_coarsened_wf>(
            velocity_verlet_wf(timestep, mass, velocities, dim_x, dim_y, dim_z),
            positions_view,
            forces_view,
            new_positions_view);
    
    // Copy new positions to positions array
    for (size_t i = 0; i < num_particles; i++) {
      positions_array[i] = new_positions_array[i];
    }
    
    // Recalculate forces at new positions
    stapl::map_func<stapl::skeletons::tags::with_coarsened_wf>(
            force_calculation_wf(sigma6, sigma12, epsilon, cutoff_sq, dim_x, dim_y, dim_z),
            positions_view,
            forces_view);

    // Get forces for the velocity update
    vector<force_vector_t> new_forces(num_particles);
    for (size_t i = 0; i < num_particles; i++) {
      new_forces[i] = forces_array[i];
    }
    
    // Complete velocity update
    for (size_t i = 0; i < num_particles; i++) {
      velocities[i].x += 0.5 * new_forces[i].x / mass * timestep;
      velocities[i].y += 0.5 * new_forces[i].y / mass * timestep;
      velocities[i].z += 0.5 * new_forces[i].z / mass * timestep;
    }
    
    // Optional: Calculate and report energy at intervals
    if (step % 100 == 0) {
      double energy =
          stapl::map_reduce<stapl::skeletons::tags::with_coarsened_wf>(
              energy_calculation_wf(sigma, epsilon, cutoff, mass, velocities, dim_x, dim_y, dim_z),
              std::plus<double>(),
              positions_view);
      
      stapl::do_once([step, energy] {
        std::cout << "Step " << step << ", Energy: " << energy << std::endl;
      });
    }
    
    execution_times[step] = timer.stop();
    timer.reset();
    timer.start();
  }
  
  // Report timing results
  report_result("Lennard-Jones MD", "STAPL", true, execution_times);
  double total = std::accumulate(execution_times.begin(),
                               execution_times.end(), 0.0);
  std::cout << "Total sim time: " << total << " s\n";
  

  return EXIT_SUCCESS;
}