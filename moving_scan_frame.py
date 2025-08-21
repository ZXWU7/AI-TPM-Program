

    def generate_systematic_grid_positions(
        self, center_position=(0, 0), scan_area_size=None, grid_spacing=None, ratio=0.9
    ):
        """
        Generate a 2D systematic cubic grid of scan positions covering the specified area.

        Args:
            center_position: (x, y) center position in meters, default is (0, 0)
            scan_area_size: size of the scan area in meters, default uses zoom_out_scale
            grid_spacing: spacing between grid points in meters, default uses Scan_edge

        Returns:
            list: List of (x, y) coordinates in meters representing scan positions
        """
        # Use default values if not provided
        if scan_area_size is None:
            scan_area_size = self.convert(
                self.zoom_out_scale
            )  # Convert "100n" to meters

        if grid_spacing is None:
            grid_spacing = ratio * self.convert(
                self.scan_zoom_in_list[0]
            )  # Convert "20n" to meters

        # Calculate the number of grid points in each direction
        # Ensure we have odd numbers to include the center point
        num_points_x = int(scan_area_size / grid_spacing)
        num_points_y = int(scan_area_size / grid_spacing)

        # Make sure we have odd numbers for symmetric grid around center
        if num_points_x % 2 == 0:
            num_points_x += 1
        if num_points_y % 2 == 0:
            num_points_y += 1

        # Calculate half-ranges for symmetric grid
        half_range_x = (num_points_x - 1) // 2
        half_range_y = (num_points_y - 1) // 2

        # Generate grid positions
        grid_positions = []

        for i in range(-half_range_x, half_range_x + 1):
            for j in range(-half_range_y, half_range_y + 1):
                x_pos = center_position[0] + i * grid_spacing
                y_pos = center_position[1] + j * grid_spacing
                grid_positions.append((x_pos, y_pos))

        print(
            f"  - Scan area: {scan_area_size*1e9:.1f} nm x {scan_area_size*1e9:.1f} nm"
        )
        print(f"  - Grid size: {num_points_x} x {num_points_y}")

        return grid_positions

    def generate_optimized_scan_path(self, grid_positions):
        """
        Generate an optimized scan path through the grid positions to minimize travel distance.
        Uses a simple zigzag pattern for efficient scanning.

        Args:
            grid_positions: list of (x, y) coordinates in meters

        Returns:
            list: Optimized sequence of (x, y) coordinates
        """
        if not grid_positions:
            return []

        # Sort positions by y-coordinate first, then by x-coordinate
        # This creates rows of points
        sorted_positions = sorted(grid_positions, key=lambda pos: (pos[1], pos[0]))

        # Group by y-coordinate (rows)
        rows = {}
        for pos in sorted_positions:
            y = pos[1]
            if y not in rows:
                rows[y] = []
            rows[y].append(pos)

        # Create zigzag pattern: alternate direction for each row
        optimized_path = []
        reverse_row = False

        for y in sorted(rows.keys()):
            row_positions = sorted(rows[y], key=lambda pos: pos[0])
            if reverse_row:
                row_positions.reverse()
            optimized_path.extend(row_positions)
            reverse_row = not reverse_row

        print(f"Optimized scan path with {len(optimized_path)} positions")
        print(f"  - Pattern: Zigzag rows for minimal travel distance")

        return optimized_path

    def move_to_grid_position(self, position_index=0, grid_positions=None): ######################################## insert into main program
        """
        Move to a specific position in the systematic grid.

        Args:
            position_index: index of the position in the grid list
            grid_positions: list of grid positions, if None uses stored grid

        Returns:
            tuple: (x, y) position in meters where the tip moved
        """
        if grid_positions is None:
            if not hasattr(self, "systematic_grid_positions"):
                raise ValueError("No grid positions available. Generate grid first.")
            grid_positions = self.systematic_grid_positions

        if position_index >= len(grid_positions):
            raise ValueError(
                f"Position index {position_index} out of range (max: {len(grid_positions)-1})"
            )

        target_position = grid_positions[position_index]

        # Update coordinate tracking
        self.nanocoodinate = target_position
        self.nanocoodinate_list.append(self.nanocoodinate)

        print(
            f"Moving to grid position {position_index+1}/{len(grid_positions)}: "
            f"({target_position[0]*1e9:.1f}, {target_position[1]*1e9:.1f}) nm"
        )

        # Set scan frame at the target position
        self.ScanFrameSet(
            target_position[0],
            target_position[1] + 0.5 * self.Scan_edge_SI,
            self.Scan_edge,
            self.Scan_edge,
            angle=0,
        ) #################################################

        # Save updated coordinate list
        np.save(self.log_path + "/nanocoodinate_list.npy", self.nanocoodinate_list)

        return target_position

    def initialize_systematic_scanning(
        self, center_position=(0, 0), scan_area_size=None, grid_spacing=None
    ):
        """
        Initialize systematic 2D grid scanning with optimized path.

        Args:
            center_position: (x, y) center position in meters
            scan_area_size: size of the scan area in meters (default: zoom_out_scale)
            grid_spacing: spacing between grid points in meters (default: Scan_edge)

        Returns:
            dict: Information about the generated grid and path
        """
        # Generate grid positions
        grid_positions = self.generate_systematic_grid_positions(
            center_position, scan_area_size, grid_spacing
        )

        # Optimize scan path
        optimized_path = self.generate_optimized_scan_path(grid_positions)

        # Store for later use
        self.systematic_grid_positions = optimized_path
        self.grid_scan_index = 0

        # Create visualization data
        grid_info = {
            "total_positions": len(optimized_path),
            "center_position": center_position,
            "scan_area_size": scan_area_size or self.convert(self.zoom_out_scale),
            "grid_spacing": grid_spacing or self.convert(self.Scan_edge),
            "positions": optimized_path,
        }

        print("Systematic scanning initialized:")
        print(f"  - Total scan positions: {grid_info['total_positions']}")
        print(
            f"  - Scan area: {grid_info['scan_area_size']*1e9:.1f} nm x {grid_info['scan_area_size']*1e9:.1f} nm"
        )
        print(f"  - Grid spacing: {grid_info['grid_spacing']*1e9:.1f} nm")

        return grid_info

    def move_to_next_grid_position(self):
        """
        Move to the next position in the systematic grid scan.

        Returns:
            tuple: (x, y) position in meters, or None if scan complete
        """
        if not hasattr(self, "systematic_grid_positions"):
            raise ValueError(
                "Systematic scanning not initialized. Call initialize_systematic_scanning() first."
            )

        if self.grid_scan_index >= len(self.systematic_grid_positions):
            print("Systematic grid scan completed!")
            return None

        position = self.move_to_grid_position(
            self.grid_scan_index, self.systematic_grid_positions
        )
        self.grid_scan_index += 1

        return position

    def example_systematic_scanning_usage(self):
        """
        Example method demonstrating how to use the systematic scanning functionality.
        This shows the typical workflow for setting up and using systematic grid scanning.
        """
        print("=== Systematic Scanning Example ===")

        # Example 1: Basic grid with default parameters
        print("\n1. Creating basic systematic grid...")
        grid_info = self.initialize_systematic_scanning()

        # Save configuration
        self.save_grid_configuration(grid_info)

        # Example 2: Custom grid with specific parameters
        print("\n2. Creating custom systematic grid...")
        center = (5e-9, -3e-9)  # 5nm right, 3nm down from origin
        area_size = 80e-9  # 80 nm total area
        spacing = 15e-9  # 15 nm between scan points

        custom_grid_info = self.initialize_systematic_scanning(
            center_position=center, scan_area_size=area_size, grid_spacing=spacing
        )

        return grid_info, custom_grid_info
