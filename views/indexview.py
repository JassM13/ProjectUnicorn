from fasthtml.common import *

def index_view():
    return Div(
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.7.1/gsap.min.js"),
        Style("""
            @keyframes float {
                0% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-20px) rotate(5deg); }
                100% { transform: translateY(0px) rotate(0deg); }
            }
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            .hero-text {
                color: var(--primary-color);
                font-size: 4.5em;
            }
            .cta-button {
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                font-size: 1.1em;
                letter-spacing: 0.5px;
            }
            .cta-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 20px var(--primary-dark);
            }
            .cta-button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(120deg, transparent, rgba(255,255,255,0.3), transparent);
                transition: 0.5s;
            }
            .cta-button:hover::before {
                left: 100%;
            }
            .feature-card {
                background: #222;
                border-radius: 12px;
                padding: 32px;
                transition: transform 0.3s ease;
                border: 1px solid #333;
            }
            .feature-card:hover {
                transform: translateY(-10px);
                border-color: var(--primary-light);
            }
            .feature-icon {
                font-size: 2.5em;
                color: var(--primary-light);
                margin-bottom: 20px;
            }
            /* Mobile responsiveness overrides */
            @media (max-width: 768px) {
                .main-container {
                    border-radius: 0 !important;
                    height: auto !important;
                    top: 0 !important;
                    left: 0 !important;
                    right: 0 !important;
                    bottom: 0 !important;
                    padding: 10px !important;
                }
                .hero-text {
                    font-size: 3em !important;
                }
                .cta-button {
                    font-size: 1em !important;
                    padding: 12px 30px !important;
                }
            }
        """),
        Script("""
            document.addEventListener('DOMContentLoaded', () => {
                // Seeded initialization
                const seed = localStorage.getItem('unicornSeed') || 
                    (Math.random().toString(36).substr(2, 9) + Date.now());
                localStorage.setItem('unicornSeed', seed);

                // Seeded random generator
                const seededRandom = (() => {
                    let value = 0;
                    for (let i = 0; i < seed.length; i++) {
                        value += seed.charCodeAt(i) * (i + 1);
                    }
                    return () => {
                        value = Math.sin(value) * 10000;
                        const val = value - Math.floor(value);
                        value = val * 1000;
                        return val;
                    };
                })();

                // Three.js Scene Setup
                const scene = new THREE.Scene();
                const container = document.getElementById('three-container');
                const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({ 
                    alpha: true,
                    antialias: true,
                    precision: 'highp'
                });
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.setSize(container.clientWidth, container.clientHeight);
                container.appendChild(renderer.domElement);

                // Handle window resize
                window.addEventListener('resize', () => {
                    camera.aspect = container.clientWidth / container.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, container.clientHeight);
                });

                // Create candlestick objects
                const candlesticks = [];
                const candlestickMaterial = new THREE.MeshStandardMaterial({
                    color: 0x333333,
                    metalness: 0.3,
                    roughness: 0.2,
                    envMapIntensity: 1.0
                });

                function createCandlestick() {
                    const height = 1 + seededRandom() * 2;
                    const width = 0.3;
                    const wickHeight = height * 1.5;

                    const bodyGeometry = new THREE.BoxGeometry(width, height, width, 8, 8, 8);
                    const wickGeometry = new THREE.BoxGeometry(width/3, wickHeight, width/3, 4, 8, 4);

                    const body = new THREE.Mesh(bodyGeometry, candlestickMaterial);
                    const wick = new THREE.Mesh(wickGeometry, candlestickMaterial);

                    body.castShadow = true;
                    body.receiveShadow = true;
                    wick.castShadow = true;
                    wick.receiveShadow = true;

                    const candlestick = new THREE.Group();
                    candlestick.add(body);
                    candlestick.add(wick);

                    // Original expanded spawn area with seeded randomness
                    candlestick.position.set(
                        seededRandom() * 80 - 40,  // X: -40 to +40
                        seededRandom() * 80 - 40,  // Y: -40 to +40
                        seededRandom() * 40 - 45   // Z: -45 to -5
                    );
                    
                    candlestick.rotation.x = seededRandom() * Math.PI;
                    candlestick.rotation.y = seededRandom() * Math.PI;
                    
                    scene.add(candlestick);
                    candlesticks.push({
                        object: candlestick,
                        speed: 0.005 + seededRandom() * 0.01
                    });
                }
                
                // Create initial candlesticks
                for (let i = 0; i < 50; i++) {
                    createCandlestick();
                }
                
                // Enhanced lighting setup
                const mainLight = new THREE.PointLight(0xffffff, 1.5, 100);
                mainLight.position.set(10, 10, 10);
                scene.add(mainLight);

                const fillLight = new THREE.PointLight(0xf6cd70, 0.3, 50);
                fillLight.position.set(-10, -5, -10);
                scene.add(fillLight);

                scene.add(new THREE.AmbientLight(0x404040, 0.5));

                camera.position.z = 5;

                function animate() {
                    requestAnimationFrame(animate);
                    candlesticks.forEach(candlestick => {
                        candlestick.object.rotation.x += candlestick.speed;
                        candlestick.object.rotation.y += candlestick.speed;
                    });
                    renderer.render(scene, camera);
                }
                animate();
            });
        """),
        Div(
            Div(id="three-container", style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;"),
            Div(
                H1("Project Unicorn", cls="hero-text", style="margin-bottom: 24px; font-weight: 800; letter-spacing: -1px;"),
                P("Where Innovation Meets Intelligence", style="color: var(--text-primary); font-size: 1.5em; margin-bottom: 40px; text-shadow: 0 2px 10px rgba(0,0,0,0.3);"),
                Div(
                    A("Experience the Magic", href="/login", cls="cta-button", 
                      style="background: var(--primary-light); color: var(--background-dark); text-decoration: none; padding: 16px 40px; border-radius: 30px; font-weight: bold; margin-right: 24px;"),
                    A("Join the Journey", href="/register", cls="cta-button",
                      style="background: transparent; color: var(--primary-light); text-decoration: none; padding: 15px 39px; border-radius: 30px; font-weight: bold; border: 2px solid var(--primary-light);"),
                    style="display: flex; gap: 20px; justify-content: center;"
                ),
                style="text-align: center; position: relative; z-index: 2;"
            ),
            cls="main-container",
            style="""padding: 20px; background-color: #000; color: white; height: 95vh;
                border-radius: 16px; display: flex; flex-direction: column;
                justify-content: center; align-items: center; position: absolute;
                right: 20px; top: 20px; left: 20px; bottom: 20px;"""
        )
    )
