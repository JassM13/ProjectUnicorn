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
                color: #f6cd70;
            }
            .cta-button {
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .cta-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(246, 205, 112, 0.3);
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
                border-color: #f6cd70;
            }
            .feature-icon {
                font-size: 2.5em;
                color: #f6cd70;
                margin-bottom: 20px;
            }
        """),
        Script("""
            document.addEventListener('DOMContentLoaded', () => {
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
                    const height = Math.random() * 2 + 1;
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
                    
                    // Expanded spawn area
                    candlestick.position.x = Math.random() * 80 - 40;
                    candlestick.position.y = Math.random() * 80 - 40;
                    candlestick.position.z = Math.random() * 40 - 45;
                    
                    candlestick.rotation.x = Math.random() * Math.PI;
                    candlestick.rotation.y = Math.random() * Math.PI;
                    
                    scene.add(candlestick);
                    candlesticks.push({
                        object: candlestick,
                        speed: Math.random() * 0.01 + 0.005
                    });
                }
                
                // Create more initial candlesticks for better coverage
                for (let i = 0; i < 50; i++) {
                    createCandlestick();
                }
                
                // Remove the interval that creates new candlesticks
                // setInterval(createCandlestick, 3000);

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

                // Particle effect
                // function createParticle() {
                //     const particle = document.createElement('div');
                //     particle.className = 'particle';
                //     particle.style.width = '8px';
                //     particle.style.height = '8px';
                //     particle.style.background = '#333333';
                //     particle.style.borderRadius = '50%';
                //     particle.style.left = Math.random() * 100 + 'vw';
                //     particle.style.top = Math.random() * 100 + 'vh';
                //     document.body.appendChild(particle);
                //     setTimeout(() => particle.remove(), 6000);
                // }

                // setInterval(createParticle, 200);
            });
        """),
        Div(
            Div(id="three-container", style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;"),
            Div(
                H1("Project Unicorn", cls="hero-text", style="font-size: 4.5em; margin-bottom: 24px; font-weight: 800; letter-spacing: -1px;"),
                P("Where Innovation Meets Intelligence", style="color: white; font-size: 1.5em; margin-bottom: 40px; text-shadow: 0 2px 10px rgba(0,0,0,0.3);"),
                Div(
                    A("Experience the Magic", href="/login", cls="cta-button", 
                      style="background: #f6cd70; color: #000; text-decoration: none; \
                             padding: 16px 40px; border-radius: 30px; font-weight: bold; margin-right: 24px; \
                             font-size: 1.1em; letter-spacing: 0.5px;"),
                    A("Join the Journey", href="/register", cls="cta-button",
                      style="background: transparent; color: #f6cd70; text-decoration: none; padding: 15px 39px; \
                             border-radius: 30px; font-weight: bold; border: 2px solid #f6cd70; font-size: 1.1em; \
                             letter-spacing: 0.5px;"),
                    style="display: flex; gap: 20px; justify-content: center;"
                ),
                style="text-align: center; position: relative; z-index: 2;"
            ),
            style="""padding: 20px; background-color: #000; color: white; height: 95vh;
                border-radius: 16px; display: flex; flex-direction: column;
                justify-content: center; align-items: center; position: absolute;
                right: 20px; top: 20px; left: 20px; bottom: 20px;"""
        )
    )