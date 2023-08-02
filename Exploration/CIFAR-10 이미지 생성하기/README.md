# Code Peer Review Templete
- 코더 : 김재림
- 리뷰어 : 박기용

---

# PRT(PeerReviewTemplate)

각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.

- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [0] 주석을 보고 작성자의 코드가 이해되었나요?
- [0] 코드가 에러를 유발할 가능성이 있나요?
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [0] 코드가 간결한가요?

---
# 참고 링크 및 코드 개선 여부

- 모델 개선 하시는 부분에서 수정이 필요한 부분만 리뷰 하겠습니다.
- 저도 같은 부분에서 수정을 안해서 학습이 제대로 진행이 안되었어서 조언 드리고자 합니다.
- 

```python
#개선한 model  
#generator_two = make_generator_model_two()
#discriminator_two = make_discriminator_model_two()
## 위와 같이 모델을 할당하면 아래 trainstep 에서도 함수 부분을 동일하게 설정 해주셔야 합니다.
## 그렇지 않으면 기존 모델의 가중치만 계속 업데이트 되어 버립니다.
## generator = > generator_two ,, discriminator => discriminator_two
## train 과 train_step 부분 적용 하시면 됩니다.

### 수정 예시
@tf.function
# (1) 입력 데이터
def train_step(images):
    
    # (2) 생성자 입력 노이즈
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    #(3) tf.GradientTape() 오픈
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #(4) generated_images 생성
      generated_images = generator_two(noise, training=True)

    #(5) discriminator 판별
      real_output = discriminator_two(images, training=True)
      fake_output = discriminator_two(generated_images, training=True)

    #(6) loss 계산
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
    
    #(7) accuracy 계산
      real_accuracy, fake_accuracy = discriminator_accuracy(real_output, fake_output)

    #(8) gradient 계산
    gradients_of_generator = gen_tape.gradient(gen_loss, generator_two.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_two.trainable_variables)

    #(9) 모델 학습
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_two.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_two.trainable_variables))
    
    #(10) 리턴값
    return gen_loss, disc_loss, real_accuracy, fake_accuracy

## train 에서도 수정 해주지 않으시면 이미지가 제대로 나타나지 않습니다.
def train(dataset, epochs, save_every):
    start = time.time()
    history = {'gen_loss':[], 'disc_loss':[], 'real_accuracy':[], 'fake_accuracy':[]}

    for epoch in range(epochs):
        epoch_start = time.time()
        for it, image_batch in enumerate(dataset):
            gen_loss, disc_loss, real_accuracy, fake_accuracy = train_step(image_batch)
            history['gen_loss'].append(gen_loss)
            history['disc_loss'].append(disc_loss)
            history['real_accuracy'].append(real_accuracy)
            history['fake_accuracy'].append(fake_accuracy)

            if it % 50 == 0:
                display.clear_output(wait=True)
                generate_and_save_images(generator_two, epoch+1, it+1, seed)
                print('Epoch {} | iter {}'.format(epoch+1, it+1))
                print('Time for epoch {} : {} sec'.format(epoch+1, int(time.time()-epoch_start)))

        if (epoch + 1) % save_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        display.clear_output(wait=True)
        generate_and_save_images(generator_two, epochs, it, seed)
        print('Time for training : {} sec'.format(int(time.time()-start)))

        draw_train_history(history, epoch)
```

