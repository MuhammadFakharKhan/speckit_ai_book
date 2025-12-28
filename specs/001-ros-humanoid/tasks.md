# Tasks: Fix Docusaurus Build Broken Links

**Input**: Feature specification from `/specs/001-ros-humanoid/spec.md`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Goal**: Fix Docusaurus build failures caused by broken links to /docs/ros2-fundamentals, /docs/python-agents, /docs/urdf-humanoids that appear site-wide in navbar/footer/sidebar

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Analyze current Docusaurus site structure and identify all broken link locations
- [X] T002 [P] Create missing documentation directories: site/docs/ros2-fundamentals, site/docs/python-agents, site/docs/urdf-humanoids
- [X] T003 [P] Document current build failure logs and error messages

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core fixes that MUST be complete before site can build successfully

**⚠️ CRITICAL**: Site cannot build until these broken links are resolved

- [X] T004 Update sidebar.js to use correct paths for missing documentation (module1/ros2-fundamentals instead of ros2-fundamentals)
- [X] T005 [P] Update docusaurus.config.js navbar links to use valid paths or remove broken links temporarily
- [X] T006 [P] Update docusaurus.config.js footer links to use valid paths or remove broken links temporarily
- [X] T007 Configure Docusaurus to not throw on broken links during development (temporary setting)

---
## Phase 3: [US1] Create ROS2 Fundamentals Documentation

**Goal**: Create the missing ROS2 fundamentals documentation to satisfy the broken link requirement

**Independent Test**: Can navigate to /docs/ros2-fundamentals and view complete content

### Implementation for User Story 1

- [X] T008 [P] [US1] Create ROS2 fundamentals content with learning objectives in site/docs/module1/ros2-fundamentals.md
- [X] T009 [US1] Add ROS2 nodes, topics, services, and actions explanations to the fundamentals document
- [X] T010 [US1] Include runnable ROS2 Python examples for joint commands in the fundamentals document
- [X] T011 [US1] Add proper Docusaurus frontmatter to the fundamentals document
- [X] T012 [US1] Update sidebar to properly reference the ROS2 fundamentals document

**Checkpoint**: At this point, /docs/ros2-fundamentals should be accessible and build successfully

---
## Phase 4: [US2] Create Python Agents Documentation

**Goal**: Create the missing Python agents documentation to satisfy the broken link requirement

**Independent Test**: Can navigate to /docs/python-agents and view complete content

### Implementation for User Story 2

- [X] T013 [P] [US2] Create Python agents content with learning objectives in site/docs/python-agents.md
- [X] T014 [US2] Add rclpy integration examples and explanations to the Python agents document
- [X] T015 [US2] Include examples of Python AI agents interfacing with ROS2 in the document
- [X] T016 [US2] Add proper Docusaurus frontmatter to the Python agents document
- [X] T017 [US2] Update footer and navbar to properly reference the Python agents document

**Checkpoint**: At this point, /docs/python-agents should be accessible and build successfully

---
## Phase 5: [US3] Create URDF Humanoids Documentation

**Goal**: Create the missing URDF humanoids documentation to satisfy the broken link requirement

**Independent Test**: Can navigate to /docs/urdf-humanoids and view complete content

### Implementation for User Story 3

- [X] T018 [P] [US3] Create URDF humanoids content with learning objectives in site/docs/urdf-humanoids.md
- [X] T019 [US3] Add URDF design principles and examples for humanoid robots to the document
- [X] T020 [US3] Include visualization and model loading examples in the URDF document
- [X] T021 [US3] Add proper Docusaurus frontmatter to the URDF humanoids document
- [X] T022 [US3] Update footer and navbar to properly reference the URDF humanoids document

**Checkpoint**: At this point, /docs/urdf-humanoids should be accessible and build successfully

---
## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and optimization of all fixes

- [X] T023 [P] Test Docusaurus build locally to ensure no broken links remain
- [X] T024 Run full site build to verify all links resolve correctly
- [X] T025 [P] Update onBrokenLinks setting back to 'throw' in docusaurus.config.js
- [X] T026 Verify all navigation elements work correctly in development and production builds
- [X] T027 [P] Test site navigation across different pages to ensure no regressions
- [X] T028 Deploy updated site to staging environment for review

---
## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS successful builds
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - No dependencies on other stories

### Within Each User Story

- Content creation before navigation updates
- Proper frontmatter required for Docusaurus integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Different user stories can be worked on in parallel by different team members

---
## Implementation Strategy

### MVP First (Foundational Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks successful builds)
3. **STOP and VALIDATE**: Test that Docusaurus builds without broken link errors
4. Site should now build successfully but with placeholder/missing content

### Incremental Delivery

1. Complete Setup + Foundational → Site builds successfully (no broken links)
2. Add User Story 1 → ROS2 fundamentals content → Test and verify
3. Add User Story 2 → Python agents content → Test and verify
4. Add User Story 3 → URDF humanoids content → Test and verify
5. Each story adds complete, functional content

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (ROS2 fundamentals)
   - Developer B: User Story 2 (Python agents)
   - Developer C: User Story 3 (URDF humanoids)
3. Stories complete and integrate independently