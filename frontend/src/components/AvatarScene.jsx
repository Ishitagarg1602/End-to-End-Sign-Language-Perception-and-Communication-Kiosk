import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

/* ─── Programmatic ISL Robot Avatar ───
   A unique geometric humanoid built entirely in Three.js.
   It has articulated arms, hands with fingers, and a head with
   an expressive "face". Gestures are driven by keyword mapping.
*/

// ── Keyword → gesture mapping ──
const GESTURE_MAP = {
  wait: 'idle', moment: 'idle', patience: 'idle',
  hello: 'wave', welcome: 'wave', hi: 'wave',
  please: 'point', show: 'point', sign: 'point', look: 'point',
  complete: 'thumbsup', done: 'thumbsup', thank: 'thumbsup', okay: 'thumbsup', ok: 'thumbsup',
  yes: 'nod', agree: 'nod', correct: 'nod', right: 'nod',
  no: 'shake', wrong: 'shake', sorry: 'shake',
  id: 'point', document: 'point', card: 'point',
};

export function getGestureForText(text) {
  if (!text) return 'talk';
  const lower = text.toLowerCase();
  for (const [keyword, gesture] of Object.entries(GESTURE_MAP)) {
    if (lower.includes(keyword)) return gesture;
  }
  return 'talk';
}

// ── Robot Body Component ──
function RobotAvatar({ gesture = 'idle', isActive = false }) {
  const groupRef = useRef();
  const leftArmRef = useRef();
  const rightArmRef = useRef();
  const leftForearmRef = useRef();
  const rightForearmRef = useRef();
  const headRef = useRef();
  const eyeLeftRef = useRef();
  const eyeRightRef = useRef();
  const mouthRef = useRef();

  // Materials
  const bodyMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#2563EB', metalness: 0.3, roughness: 0.4
  }), []);
  const accentMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#60A5FA', metalness: 0.5, roughness: 0.3
  }), []);
  const darkMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1E3A5F', metalness: 0.4, roughness: 0.5
  }), []);
  const eyeMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#FFFFFF', emissive: '#FFFFFF', emissiveIntensity: 0.3
  }), []);
  const pupilMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1a1a2e', emissive: '#3B82F6', emissiveIntensity: 0.5
  }), []);
  const mouthMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#60A5FA', emissive: '#60A5FA', emissiveIntensity: 0.4
  }), []);
  const handMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#93C5FD', metalness: 0.3, roughness: 0.4
  }), []);

  useFrame((frameState) => {
    if (!isActive) {
      // Gentle idle breathing
      if (groupRef.current) {
        groupRef.current.position.y = Math.sin(frameState.clock.elapsedTime * 0.8) * 0.02;
      }
      // Reset arms
      if (leftArmRef.current) leftArmRef.current.rotation.z = 0.15;
      if (rightArmRef.current) rightArmRef.current.rotation.z = -0.15;
      if (leftForearmRef.current) leftForearmRef.current.rotation.x = 0;
      if (rightForearmRef.current) rightForearmRef.current.rotation.x = 0;
      if (headRef.current) { headRef.current.rotation.y = 0; headRef.current.rotation.x = 0; }
      if (mouthRef.current) mouthRef.current.scale.y = 1;
      return;
    }

    const t = frameState.clock.elapsedTime;

    // Breathing
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(t * 1.2) * 0.03;
    }

    switch (gesture) {
      case 'wave': {
        // Right arm waves
        if (rightArmRef.current) {
          rightArmRef.current.rotation.z = -1.4;
          rightArmRef.current.rotation.x = 0;
        }
        if (rightForearmRef.current) {
          rightForearmRef.current.rotation.z = Math.sin(t * 4) * 0.5;
        }
        if (leftArmRef.current) leftArmRef.current.rotation.z = 0.15;
        if (headRef.current) headRef.current.rotation.z = Math.sin(t * 2) * 0.08;
        if (mouthRef.current) mouthRef.current.scale.y = 1.3;
        break;
      }
      case 'point': {
        // Right arm points forward
        if (rightArmRef.current) {
          rightArmRef.current.rotation.z = -0.8;
          rightArmRef.current.rotation.x = -0.6;
        }
        if (rightForearmRef.current) {
          rightForearmRef.current.rotation.x = -0.3;
        }
        if (leftArmRef.current) leftArmRef.current.rotation.z = 0.1;
        if (headRef.current) headRef.current.rotation.y = Math.sin(t * 1.5) * 0.1;
        if (mouthRef.current) mouthRef.current.scale.y = 1;
        break;
      }
      case 'thumbsup': {
        // Right arm up, fist with thumb
        if (rightArmRef.current) {
          rightArmRef.current.rotation.z = -1.0;
          rightArmRef.current.rotation.x = -0.3;
        }
        if (rightForearmRef.current) {
          rightForearmRef.current.rotation.x = -0.8;
        }
        if (leftArmRef.current) leftArmRef.current.rotation.z = 0.1;
        if (headRef.current) headRef.current.rotation.x = -0.1; // slight nod
        if (mouthRef.current) mouthRef.current.scale.y = 1.5; // smile
        break;
      }
      case 'nod': {
        if (headRef.current) {
          headRef.current.rotation.x = Math.sin(t * 3) * 0.2;
        }
        if (leftArmRef.current) leftArmRef.current.rotation.z = 0.15;
        if (rightArmRef.current) rightArmRef.current.rotation.z = -0.15;
        if (mouthRef.current) mouthRef.current.scale.y = 1.2;
        break;
      }
      case 'shake': {
        if (headRef.current) {
          headRef.current.rotation.y = Math.sin(t * 4) * 0.25;
        }
        if (leftArmRef.current) leftArmRef.current.rotation.z = 0.3;
        if (rightArmRef.current) rightArmRef.current.rotation.z = -0.3;
        if (mouthRef.current) mouthRef.current.scale.y = 0.5;
        break;
      }
      case 'talk':
      default: {
        // Both arms gesture expressively (like explaining)
        if (leftArmRef.current) {
          leftArmRef.current.rotation.z = 0.6 + Math.sin(t * 2.5) * 0.3;
          leftArmRef.current.rotation.x = Math.sin(t * 1.8) * 0.2;
        }
        if (rightArmRef.current) {
          rightArmRef.current.rotation.z = -0.6 - Math.sin(t * 2.5 + 1) * 0.3;
          rightArmRef.current.rotation.x = Math.sin(t * 1.8 + 1) * 0.2;
        }
        if (leftForearmRef.current) {
          leftForearmRef.current.rotation.x = Math.sin(t * 3) * 0.3;
        }
        if (rightForearmRef.current) {
          rightForearmRef.current.rotation.x = Math.sin(t * 3 + 0.5) * 0.3;
        }
        if (headRef.current) {
          headRef.current.rotation.y = Math.sin(t * 1.5) * 0.12;
          headRef.current.rotation.x = Math.sin(t * 2) * 0.05;
        }
        // Mouth opens/closes while "talking"
        if (mouthRef.current) {
          mouthRef.current.scale.y = 0.8 + Math.abs(Math.sin(t * 6)) * 0.8;
        }
        break;
      }
    }
  });

  return (
    <group ref={groupRef} position={[0, -0.8, 0]}>
      {/* ── Torso ── */}
      <mesh material={bodyMat} position={[0, 0.5, 0]}>
        <boxGeometry args={[0.7, 0.9, 0.4]} />
      </mesh>
      {/* Chest accent */}
      <mesh material={accentMat} position={[0, 0.65, 0.21]}>
        <boxGeometry args={[0.3, 0.15, 0.02]} />
      </mesh>
      {/* Core light */}
      <mesh position={[0, 0.45, 0.21]}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshStandardMaterial color="#3B82F6" emissive="#3B82F6" emissiveIntensity={1.5} />
      </mesh>

      {/* ── Head ── */}
      <group ref={headRef} position={[0, 1.2, 0]}>
        <mesh material={bodyMat}>
          <boxGeometry args={[0.45, 0.45, 0.4]} />
        </mesh>
        {/* Visor */}
        <mesh material={darkMat} position={[0, 0.02, 0.21]}>
          <boxGeometry args={[0.38, 0.15, 0.02]} />
        </mesh>
        {/* Eyes */}
        <mesh ref={eyeLeftRef} material={eyeMat} position={[-0.1, 0.04, 0.22]}>
          <sphereGeometry args={[0.04, 12, 12]} />
        </mesh>
        <mesh position={[-0.1, 0.04, 0.24]} material={pupilMat}>
          <sphereGeometry args={[0.02, 8, 8]} />
        </mesh>
        <mesh ref={eyeRightRef} material={eyeMat} position={[0.1, 0.04, 0.22]}>
          <sphereGeometry args={[0.04, 12, 12]} />
        </mesh>
        <mesh position={[0.1, 0.04, 0.24]} material={pupilMat}>
          <sphereGeometry args={[0.02, 8, 8]} />
        </mesh>
        {/* Mouth */}
        <mesh ref={mouthRef} material={mouthMat} position={[0, -0.1, 0.21]}>
          <boxGeometry args={[0.15, 0.04, 0.02]} />
        </mesh>
        {/* Antenna */}
        <mesh material={accentMat} position={[0, 0.3, 0]}>
          <cylinderGeometry args={[0.015, 0.015, 0.12, 8]} />
        </mesh>
        <mesh position={[0, 0.38, 0]}>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshStandardMaterial color="#60A5FA" emissive="#60A5FA" emissiveIntensity={2} />
        </mesh>
      </group>

      {/* ── Left Arm ── */}
      <group ref={leftArmRef} position={[0.45, 0.8, 0]}>
        {/* Upper arm */}
        <mesh material={darkMat} position={[0.1, -0.2, 0]}>
          <boxGeometry args={[0.15, 0.4, 0.15]} />
        </mesh>
        {/* Shoulder joint */}
        <mesh material={accentMat} position={[0.05, 0, 0]}>
          <sphereGeometry args={[0.08, 12, 12]} />
        </mesh>
        {/* Forearm */}
        <group ref={leftForearmRef} position={[0.1, -0.45, 0]}>
          <mesh material={bodyMat} position={[0, -0.15, 0]}>
            <boxGeometry args={[0.12, 0.35, 0.12]} />
          </mesh>
          {/* Elbow */}
          <mesh material={accentMat}>
            <sphereGeometry args={[0.06, 10, 10]} />
          </mesh>
          {/* Hand */}
          <mesh material={handMat} position={[0, -0.38, 0]}>
            <boxGeometry args={[0.14, 0.1, 0.08]} />
          </mesh>
          {/* Fingers */}
          {[-0.04, 0, 0.04].map((x, i) => (
            <mesh key={i} material={handMat} position={[x, -0.47, 0]}>
              <boxGeometry args={[0.03, 0.08, 0.03]} />
            </mesh>
          ))}
        </group>
      </group>

      {/* ── Right Arm ── */}
      <group ref={rightArmRef} position={[-0.45, 0.8, 0]}>
        <mesh material={darkMat} position={[-0.1, -0.2, 0]}>
          <boxGeometry args={[0.15, 0.4, 0.15]} />
        </mesh>
        <mesh material={accentMat} position={[-0.05, 0, 0]}>
          <sphereGeometry args={[0.08, 12, 12]} />
        </mesh>
        <group ref={rightForearmRef} position={[-0.1, -0.45, 0]}>
          <mesh material={bodyMat} position={[0, -0.15, 0]}>
            <boxGeometry args={[0.12, 0.35, 0.12]} />
          </mesh>
          <mesh material={accentMat}>
            <sphereGeometry args={[0.06, 10, 10]} />
          </mesh>
          <mesh material={handMat} position={[0, -0.38, 0]}>
            <boxGeometry args={[0.14, 0.1, 0.08]} />
          </mesh>
          {[-0.04, 0, 0.04].map((x, i) => (
            <mesh key={i} material={handMat} position={[x, -0.47, 0]}>
              <boxGeometry args={[0.03, 0.08, 0.03]} />
            </mesh>
          ))}
        </group>
      </group>

      {/* ── Waist ── */}
      <mesh material={darkMat} position={[0, 0, 0]}>
        <boxGeometry args={[0.55, 0.15, 0.35]} />
      </mesh>

      {/* ── Left Leg ── */}
      <mesh material={darkMat} position={[0.18, -0.45, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
      </mesh>
      <mesh material={bodyMat} position={[0.18, -0.8, 0.04]}>
        <boxGeometry args={[0.2, 0.12, 0.3]} />
      </mesh>

      {/* ── Right Leg ── */}
      <mesh material={darkMat} position={[-0.18, -0.45, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
      </mesh>
      <mesh material={bodyMat} position={[-0.18, -0.8, 0.04]}>
        <boxGeometry args={[0.2, 0.12, 0.3]} />
      </mesh>
    </group>
  );
}

// ── Floating particles for ambiance ──
function Particles() {
  const count = 30;
  const ref = useRef();
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 6;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 4;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 4;
    }
    return arr;
  }, []);

  useFrame((s) => {
    if (ref.current) {
      ref.current.rotation.y = s.clock.elapsedTime * 0.03;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.03} color="#60A5FA" transparent opacity={0.4} sizeAttenuation />
    </points>
  );
}

// ── Main exported scene ──
export default function AvatarScene({ gesture = 'idle', isActive = false }) {
  return (
    <Canvas
      camera={{ position: [0, 0.4, 2.8], fov: 45 }}
      style={{ width: '100%', height: '100%', borderRadius: 16, background: 'linear-gradient(180deg, #0B1120 0%, #1A1A2E 50%, #0B1120 100%)' }}
      gl={{ antialias: true, alpha: true }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[3, 5, 2]} intensity={1.2} color="#ffffff" />
      <pointLight position={[-2, 2, 3]} intensity={0.6} color="#60A5FA" />
      <pointLight position={[2, -1, 2]} intensity={0.3} color="#8B5CF6" />

      {/* Ground plane glow */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.7, 0]}>
        <circleGeometry args={[1.2, 32]} />
        <meshStandardMaterial color="#1E3A5F" emissive="#2563EB" emissiveIntensity={0.15} transparent opacity={0.5} />
      </mesh>

      <RobotAvatar gesture={gesture} isActive={isActive} />
      <Particles />

      <OrbitControls
        enableZoom={false}
        enablePan={false}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.8}
        autoRotate={!isActive}
        autoRotateSpeed={0.5}
      />
    </Canvas>
  );
}
